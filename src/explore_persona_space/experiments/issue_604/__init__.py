# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ※, σ, γ, —) in scientific docstrings + labels.
"""Shared constants + helpers for issue #604 (P0 adapter SVD).

Pure-CPU linear algebra over stored LoRA adapters: exact SVD of composed
ΔW = s·B·A per (layer, module) without materializing the d_out×d_in
matrix (QR factor trick), plus the adapter inventory across the 7 reused
lines and the persona/context prompt-resolution chain used by the Phase-B
context-vector extraction.

Plan: tasks/.../604/plans/plan.md (v2). Key registered choices implemented
here:

- rsLoRA scale s = α/√r, asserted per adapter from ``adapter_config.json``
  (``use_rslora: true`` everywhere; §11).
- Comparison-space validity map (§4 Phase A): only 3584-dim singular
  vectors enter context comparisons — attn-key stack [ΔW_q;ΔW_k;ΔW_v]
  right vectors (post-input_layernorm residual input), MLP-key stack
  [ΔW_gate;ΔW_up] right vectors (post-post_attention_layernorm input),
  residual-write stack [ΔW_o|ΔW_down] left vectors (residual output).
  down_proj INPUT (18944-d MLP-hidden), o_proj INPUT (head concat), and
  q/k/v/gate/up OUTPUTS are spectra-only.
- Realized negative panels are read from artifacts, not prose: #527 from
  ``eval_results/issue_527/pair_selection.json`` (the documented pair-2
  contamination), #538/#550 via the per-pair panel fix, #474 = the other
  15 transformations, #519 from its training mix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("issue604")

SCHEMA_VERSION = "issue604_adapter_svd_v1"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HIDDEN_SIZE = 3584
N_LAYERS = 28

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PRIVATE_DATA_REPO = "superkaiba1/explore-persona-space-data-private"
HF_BUCKET = "issue604_adapter_svd"

# Registered layer band for the key/rotation reads (plan §4 C4, §13).
KEY_LAYER_BAND = tuple(range(14, 25))  # L14–L24 inclusive

# Measured-write extraction layers per line (plan §4 C3).
EXTRACT_LAYER_DIAL = 20
EXTRACT_LAYER_519 = 14

# Top-k singular components persisted per stack for the Phase-C reads.
# Top-1/top-2 carry the key/write reads; the truncated top-8 input basis
# (V8, S) is what the Wang-et-al. constancy read projects contexts through
# (cos between rank-8 Δout vectors needs only S·V8ᵀ·v — U cancels by
# orthonormality). Truncation energy fraction is persisted alongside so
# the analyzer can qualify the approximation.
TOP_K_VECTORS = 8

# Module groups per adapter family.
ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")

# Sub-spaces valid for residual comparison (plan §4 comparison-validity map).
RESIDUAL_INPUT_MODULES = ("q_proj", "k_proj", "v_proj")  # read post-input_layernorm
RESIDUAL_INPUT_MLP_MODULES = ("gate_proj", "up_proj")  # read post-post_attention_layernorm
RESIDUAL_OUTPUT_MODULES = ("o_proj", "down_proj")  # write into the residual stream

DIAL_PAIRS = ("florist__medical_doctor", "librarian__police_officer")
DIAL_ARMS = ("A_only", "B_only", "joint")
DIAL_SEEDS = (42, 137, 256)
DIAL_ISSUES = (527, 550, 538)

# Realized negative panels (artifacts of record, plan §4 C4(a) + footer (7)):
# #527 used the pre-fix panel for BOTH pairs (pair_selection.json
# `negative_panel`); pair-2 cells are therefore panel-contaminated
# (librarian is simultaneously pair-2 source A and a negative).
DIAL_PANEL_527 = ("assistant", "librarian", "programmer", "chef")
# #550/#538 resolve per pair (negative_panel_for_pair fix).
DIAL_PANEL_PAIR1 = ("assistant", "librarian", "programmer", "chef")
DIAL_PANEL_PAIR2 = ("assistant", "kindergarten_teacher", "programmer", "chef")

I474_SOURCES = (
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1",
    "D1", "D2", "D3", "D4", "D5",
)  # fmt: skip
I474_EPOCHS = (1, 2, 3, 5)
I474_ARMS = ("pos", "loc")

I519_SEEDS = (42, 137, 256)
I519_SOURCE = "medical_doctor"
I519_NEGATIVES = ("assistant", "comedian", "police_officer", "software_engineer")

I521_SEEDS = (42, 137, 256)

I518_BEHAVIORS = ("refusal", "em")
I518_SOURCES = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)

I541_ARMS = (
    "courthouse_architecture_historian",
    "marine_biologist",
    "top_prior_wooden_furniture_carpenter",
)
I541_SEEDS = (42, 137, 256)

# The #551/#552 L14 panel (manifest of record:
# eval_results/issue_552/.../shifts_manifests/same_marker_seed42.manifest.json).
I551_PANEL_14 = (
    "assistant",
    "biographer",
    "comedian",
    "data_scientist",
    "french_person",
    "kindergarten_teacher",
    "librarian",
    "local_historian",
    "marine_biologist",
    "medical_doctor",
    "police_officer",
    "software_engineer",
    "villain",
    "zelthari_scholar",
)

# ── Vendored prompt augments ────────────────────────────────────────────────
# Provenance: src/explore_persona_space/experiments/issue_538/persona_registry.py
# on the issue-538 branch (`_AUGMENT_PERSONAS_FOR_311_PANEL`) — vendored
# verbatim because that package is not on main and Phase B runs from this
# branch's checkout.
AUGMENT_PERSONAS_311 = {
    "paramedic": "You are a paramedic who provides emergency medical care.",
    "poet": "You are a poet who writes verse and reflects on the human condition.",
    "navy_seal": "You are a Navy SEAL trained in special operations and combat.",
    "army_medic": "You are an army medic who treats wounded soldiers in combat zones.",
    "cybersec_consultant": (
        "You are a cybersecurity consultant who audits and protects digital systems."
    ),
    "pentester": "You are a penetration tester who finds and reports security vulnerabilities.",
    "private_investigator": (
        "You are a private investigator who solves cases through observation and interviews."
    ),
}

# Provenance: src/explore_persona_space/experiments/i509_syco_conditions.py on
# the issue-518 branch (`_SYCO_PERSONA_PROMPTS`, the 6 EM_SOURCES subset) —
# vendored verbatim (re-extracted via AST from the branch at implementation
# time); smoke prints them for the eyeball check (plan assumption 10).
I518_SOURCE_PROMPTS = {
    "assistant": "You are a helpful assistant.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "qwen_default": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
}


# ── Reproducibility metadata ────────────────────────────────────────────────


def git_commit(project_root: Path) -> str:
    """Short git commit hash of the repo this code runs from."""
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def result_metadata(project_root: Path, extra: dict | None = None) -> dict:
    """Reproducibility metadata block for every output JSON (CLAUDE.md rule)."""
    import torch

    meta = {
        "task": 604,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(project_root),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        # str(): TorchVersion objects poison weights_only=True re-loads of any
        # torch.save'd payload that embeds this meta block.
        "torch_version": str(torch.__version__),
        "base_model": BASE_MODEL,
        "argv": sys.argv[1:],
    }
    if extra:
        meta.update(extra)
    return meta


def sha256_text(text: str) -> str:
    """Hex sha256 of a UTF-8 string (prompt provenance fingerprints)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SEED_SUFFIX_RE = re.compile(r"_+seed\d+$")


def seed_group_key(cell_id: str) -> str:
    """Cell-id with the trailing seed token stripped — the cross-seed group key.

    Handles BOTH separator forms in the inventory: dial ``…__seed42`` and the
    single-underscore ``marker_seed42`` / ``em_turner_seed42`` /
    ``<arm>_seed42`` (#519/#521/#541). Ids without a trailing seed token
    (e.g. #474 ``pos_A1_ep1``) are returned unchanged.
    """
    return _SEED_SUFFIX_RE.sub("", cell_id)


# ── Exact SVD of composed s·B·A (never materializes d_out×d_in) ─────────────


def compose_svd(
    A: np.ndarray, B: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact thin SVD of ``scale * B @ A`` via QR of the factors.

    A: (r, d_in) LoRA-A; B: (d_out, r) LoRA-B. Returns (U, S, V) with
    U (d_out, r), S (r,), V (d_in, r) such that scale·B·A = U·diag(S)·Vᵀ.
    Cost O((d_in + d_out)·r²) — exact, not randomized (plan §11).
    """
    assert A.ndim == 2 and B.ndim == 2, (A.shape, B.shape)
    r = A.shape[0]
    assert B.shape[1] == r, f"rank mismatch: A {A.shape}, B {B.shape}"
    Qb, Rb = np.linalg.qr(B.astype(np.float64))  # (d_out, r), (r, r)
    Qa, Ra = np.linalg.qr(A.T.astype(np.float64))  # (d_in, r),  (r, r)
    Uc, S, Vct = np.linalg.svd(scale * (Rb @ Ra.T))  # r×r core — exact
    U = Qb @ Uc
    V = Qa @ Vct.T
    assert U.shape == (B.shape[0], r) and V.shape == (A.shape[1], r), (U.shape, V.shape)
    return U, S, V


def stack_block_factors(
    blocks: list[tuple[np.ndarray, np.ndarray]], mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build block factors for a stacked SVD (plan §4 Phase A).

    mode="row":  row-stack [ΔW₁; ΔW₂; …] over a SHARED input space —
        returns (A_stack (kr, d_in_shared) via vstack of As,
                 B_blockdiag ((Σd_out), kr)).
    mode="col":  column-concat [ΔW₁ | ΔW₂ | …] over a SHARED output space —
        returns (A_blockdiag (kr, Σd_in), B_cat (d_out_shared, kr)).

    The returned pair feeds ``compose_svd(A_stack, B_stack, scale)`` — the
    rsLoRA scale is applied ONCE there (shared across an adapter's modules).
    """
    assert mode in ("row", "col"), mode
    ranks = [a.shape[0] for a, _ in blocks]
    if mode == "row":
        d_in = blocks[0][0].shape[1]
        assert all(a.shape[1] == d_in for a, _ in blocks), "row-stack needs a shared input dim"
        A_stack = np.vstack([a for a, _ in blocks])
        total_out = sum(b.shape[0] for _, b in blocks)
        B_blk = np.zeros((total_out, sum(ranks)), dtype=np.float64)
        ro, co = 0, 0
        for (_, b), r in zip(blocks, ranks, strict=True):
            B_blk[ro : ro + b.shape[0], co : co + r] = b
            ro += b.shape[0]
            co += r
        return A_stack, B_blk
    d_out = blocks[0][1].shape[0]
    assert all(b.shape[0] == d_out for _, b in blocks), "col-concat needs a shared output dim"
    B_cat = np.hstack([b for _, b in blocks])
    total_in = sum(a.shape[1] for a, _ in blocks)
    A_blk = np.zeros((sum(ranks), total_in), dtype=np.float64)
    ro, co = 0, 0
    for (a, _), r in zip(blocks, ranks, strict=True):
        A_blk[ro : ro + r, co : co + a.shape[1]] = a
        ro += r
        co += a.shape[1]
    return A_blk, B_cat


def spectrum_metrics(S: np.ndarray) -> dict:
    """Top-1 energy σ₁²/Σσ², effective rank exp(−Σp·ln p), ‖·‖_F from σ."""
    s2 = np.asarray(S, dtype=np.float64) ** 2
    total = float(s2.sum())
    if total <= 0:
        return {"top1_energy": 0.0, "effective_rank": 0.0, "fro_norm": 0.0, "n_sv": int(S.size)}
    p = s2 / total
    p_nz = p[p > 0]
    eff_rank = float(np.exp(-(p_nz * np.log(p_nz)).sum()))
    return {
        "top1_energy": float(s2[0] / total),
        "effective_rank": eff_rank,
        "fro_norm": float(math.sqrt(total)),
        "n_sv": int(S.size),
    }


def rslora_scale(config: dict) -> float:
    """Assert rsLoRA + gauge validity and return s = α/√r (plan §11)."""
    assert config.get("use_rslora") is True, (
        f"adapter is not rsLoRA (use_rslora={config.get('use_rslora')!r}) — "
        "the rsLoRA alpha/sqrt(r) scale would be wrong; refusing to proceed"
    )
    targets = set(config.get("target_modules") or [])
    assert not targets & {"lm_head", "embed_tokens"}, (
        f"gauge violation: adapter targets {sorted(targets)} touch the unembedding"
    )
    assert not config.get("modules_to_save"), (
        f"gauge violation: modules_to_save={config.get('modules_to_save')!r}"
    )
    r = int(config["r"])
    alpha = float(config["lora_alpha"])
    return alpha / math.sqrt(r)


# ── Adapter inventory ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class AdapterCell:
    """One adapter cell of the §5 inventory."""

    line: str  # dial527 | dial550 | dial538 | i474 | i519 | i521 | i518 | i541
    cell_id: str  # unique within the line; filesystem-safe
    repo_id: str
    subfolder: str  # HF path prefix holding adapter_config.json + safetensors
    source_personas: tuple[str, ...]  # trained source context name(s); () for i521
    negative_personas: tuple[str, ...]  # realized contrastive panel; () if none
    seed: int | None = None
    arm: str | None = None  # dial arm / i474 arm / i518 behavior / i541 arm
    epoch: int | None = None  # i474 only
    tags: tuple[str, ...] = field(default_factory=tuple)  # e.g. panel-contaminated


def _dial_negatives(issue: int, pair: str) -> tuple[str, ...]:
    if issue == 527:
        return DIAL_PANEL_527
    return DIAL_PANEL_PAIR1 if pair == DIAL_PAIRS[0] else DIAL_PANEL_PAIR2


def _dial_sources(pair: str, arm: str) -> tuple[str, ...]:
    a, b = pair.split("__")
    if arm == "A_only":
        return (a,)
    if arm == "B_only":
        return (b,)
    return (a, b)


def build_inventory(  # noqa: C901 — one enumeration block per adapter line; flattening would scatter the inventory
    lines: list[str],
    *,
    include_dial_checkpoints: bool = False,
    model_repo_files: list[str] | None = None,
) -> list[AdapterCell]:
    """Enumerate the §5/§10 adapter inventory for the requested lines.

    ``model_repo_files`` (a ``list_repo_files`` listing of the model repo)
    is required for the #474 dual-layout resolver and the dial
    checkpoint enumeration; cells whose folder is absent from the listing
    are SKIPPED with a logged ``N/A — not stored`` (plan §8), never
    fabricated.
    """
    cells: list[AdapterCell] = []
    listing = set(model_repo_files or [])

    def _stored(repo: str, subfolder: str) -> bool:
        if repo != HF_MODEL_REPO or not listing:
            return True  # overflow repo / no listing: assume per Hub-verified plan
        return f"{subfolder}/adapter_config.json" in listing

    for issue in DIAL_ISSUES:
        line = f"dial{issue}"
        if line not in lines and "dial" not in lines:
            continue
        for pair in DIAL_PAIRS:
            for arm in DIAL_ARMS:
                for seed in DIAL_SEEDS:
                    slug = f"{pair}__{arm}__seed{seed}"
                    sub = f"adapters/issue_{issue}/{slug}"
                    tags: tuple[str, ...] = ()
                    if issue == 527 and pair == DIAL_PAIRS[1]:
                        tags = ("panel-contaminated",)
                    base = AdapterCell(
                        line=line,
                        cell_id=slug,
                        repo_id=HF_MODEL_REPO,
                        subfolder=sub,
                        source_personas=_dial_sources(pair, arm),
                        negative_personas=_dial_negatives(issue, pair),
                        seed=seed,
                        arm=arm,
                        tags=tags,
                    )
                    if _stored(HF_MODEL_REPO, sub):
                        cells.append(base)
                    else:
                        logger.warning("N/A — not stored: %s %s", line, sub)
                    if include_dial_checkpoints and listing:
                        ckpts = sorted(
                            {
                                f.split("/")[3]
                                for f in listing
                                if f.startswith(f"{sub}/checkpoint-")
                                and f.endswith("/adapter_config.json")
                            }
                        )
                        for ck in ckpts:
                            cells.append(
                                AdapterCell(
                                    line=line,
                                    cell_id=f"{slug}__{ck}",
                                    repo_id=HF_MODEL_REPO,
                                    subfolder=f"{sub}/{ck}",
                                    source_personas=base.source_personas,
                                    negative_personas=base.negative_personas,
                                    seed=seed,
                                    arm=arm,
                                    tags=(*tags, "checkpoint-intermediate"),
                                )
                            )

    if "i474" in lines:
        # The dual-layout resolver is meaningless without a real listing:
        # _stored() returns True for everything on an empty listing, which
        # would make BOTH layouts claim stored for every cell.
        assert listing, "i474 dual-layout resolver requires model_repo_files (Hub listing)"

        def _stored_full(sub: str) -> bool:
            # plan §10: a layout "resolves" only with config + safetensors.
            return (
                f"{sub}/adapter_config.json" in listing
                and f"{sub}/adapter_model.safetensors" in listing
            )

        for arm in I474_ARMS:
            for src in I474_SOURCES:
                for ep in I474_EPOCHS:
                    nested = f"adapters/i474_{arm}_{src}/_upload_ep{ep}"
                    flat = f"adapters/i474_{arm}_{src}_ep{ep}"
                    # The FLAT layout is the artifact of record: the #474
                    # trainer's registered per-epoch persist targets it
                    # (i474_phase23_train.py `path_in_repo = adapters/
                    # i474_{arm}_{cid}_ep{N}`), the #474 phase-4 eval that
                    # produced the registered exposure covariate read it, and
                    # the #549/#560 reuses read it. The NESTED `_upload_ep{N}`
                    # copies are accidental Hub duplicates of the local
                    # staging dirs (84/128 cells at the pinned revision, with
                    # DIFFERENT safetensors blobs for some cells) — never
                    # silently analyzed. Plan §10's "try nested first" order
                    # is superseded by this provenance evidence; the plan's
                    # intent ("assert exactly one resolves" = never pick
                    # among ambiguous layouts silently) is enforced as
                    # canonical-flat-or-fail.
                    if not _stored_full(flat):
                        raise AssertionError(
                            f"i474 {arm} {src} ep{ep}: canonical flat layout missing "
                            f"({flat!r}); nested staging duplicate "
                            f"{'present' if _stored_full(nested) else 'absent'} "
                            f"({nested!r}) — refusing to substitute a non-canonical copy"
                        )
                    if _stored_full(nested):
                        logger.info(
                            "i474 %s %s ep%d: nested staging duplicate ignored: %s",
                            arm,
                            src,
                            ep,
                            nested,
                        )
                    negatives = tuple(c for c in I474_SOURCES if c != src)
                    cells.append(
                        AdapterCell(
                            line="i474",
                            cell_id=f"{arm}_{src}_ep{ep}",
                            repo_id=HF_MODEL_REPO,
                            subfolder=flat,
                            source_personas=(src,),
                            # plan §4 C4(e): BOTH arms are scored on the matched
                            # source's other-15 contrast (the loc arm's realized
                            # negatives); the pos arm trained none.
                            negative_personas=negatives if arm == "loc" else (),
                            arm=arm,
                            epoch=ep,
                            seed=42,
                        )
                    )

    if "i519" in lines:
        for seed in I519_SEEDS:
            sub = f"issue_519/marker_seed{seed}"
            if _stored(HF_MODEL_REPO, sub):
                cells.append(
                    AdapterCell(
                        line="i519",
                        cell_id=f"marker_seed{seed}",
                        repo_id=HF_MODEL_REPO,
                        subfolder=sub,
                        source_personas=(I519_SOURCE,),
                        negative_personas=I519_NEGATIVES,
                        seed=seed,
                        tags=("saturated-endpoint",),
                    )
                )

    if "i521" in lines:
        for seed in I521_SEEDS:
            sub = f"adapters/issue_521/em_turner_seed{seed}/sft_narrow_adapter"
            if _stored(HF_MODEL_REPO, sub):
                cells.append(
                    AdapterCell(
                        line="i521",
                        cell_id=f"em_turner_seed{seed}",
                        repo_id=HF_MODEL_REPO,
                        subfolder=sub,
                        source_personas=(),  # no persona context — that is the point
                        negative_personas=(),
                        seed=seed,
                        tags=("em-no-source-control",),
                    )
                )

    if "i518" in lines:
        for behavior in I518_BEHAVIORS:
            for src in I518_SOURCES:
                sub = f"adapters/issue_518/{behavior}/{src}_seed42"
                if _stored(HF_MODEL_REPO, sub):
                    # source context resolution is group-aware in Phase C (the
                    # Phase-B assembly dedups byte-identical prompts and renames
                    # conflicts to ``<name>__i518_sources``).
                    cells.append(
                        AdapterCell(
                            line="i518",
                            cell_id=f"{behavior}_{src}_seed42",
                            repo_id=HF_MODEL_REPO,
                            subfolder=sub,
                            source_personas=(src,),
                            negative_personas=(),
                            seed=42,
                            arm=behavior,
                        )
                    )

    if "i541" in lines:
        for arm in I541_ARMS:
            for seed in I541_SEEDS:
                sub = f"adapters/exp541-arm_{arm}-on_policy_suppression_cn-seed{seed}"
                cells.append(
                    AdapterCell(
                        line="i541",
                        cell_id=f"{arm}_seed{seed}",
                        repo_id=HF_OVERFLOW_REPO,
                        subfolder=sub,
                        source_personas=(arm,),
                        negative_personas=(),
                        seed=seed,
                        arm=arm,
                        tags=("hot-lr-secondary",),
                    )
                )

    return cells


ALL_LINES = ("dial527", "dial550", "dial538", "i474", "i519", "i521", "i518", "i541")


def parse_lines_arg(arg: str) -> list[str]:
    """Parse ``--lines all|dial|comma,separated`` into canonical line names."""
    if arg == "all":
        return list(ALL_LINES)
    out: list[str] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok == "dial":
            out.extend(["dial527", "dial550", "dial538"])
        elif tok in ALL_LINES:
            out.append(tok)
        else:
            raise ValueError(f"unknown line {tok!r}; valid: all, dial, {', '.join(ALL_LINES)}")
    return out


# ── Persona / context prompt resolution (Phase B) ──────────────────────────


def load_persona_bank(project_root: Path) -> dict[str, str]:
    """60-persona bank. Reads the durable provenance copy first (plan §12 #16)."""
    candidates = [
        project_root / "eval_results/issue_604/provenance/persona_bank.json",
        project_root / "data/issue_472/persona_bank.json",
    ]
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text())
            personas = payload["personas"]
            assert isinstance(personas, dict) and len(personas) >= 60, len(personas)
            return dict(personas)
    raise FileNotFoundError(f"persona bank not found at any of: {candidates}")


def extract_mix_prompts(jsonl_path: Path) -> dict[str, str | None]:
    """{persona: system_prompt|None} from a training-mix JSONL.

    Handles both the dial schema (``_source`` / ``_negative_persona``) and
    the #519 schema (``persona`` + ``row_kind``). ``None`` records a row
    that carried NO system message (a genuinely bare context).
    """
    out: dict[str, str | None] = {}
    with open(jsonl_path) as f:
        for raw in f:
            row = json.loads(raw)
            name = row.get("_source") or row.get("_negative_persona") or row.get("persona")
            if name is None:
                raise KeyError(f"row in {jsonl_path.name} has no persona tag: {list(row)}")
            first = row["prompt"][0]
            sys_prompt = first["content"] if first["role"] == "system" else None
            if name in out:
                assert out[name] == sys_prompt, (
                    f"{jsonl_path.name}: persona {name!r} appears with TWO different "
                    "system prompts — the training mix is not a function of persona"
                )
            else:
                out[name] = sys_prompt
    return out
