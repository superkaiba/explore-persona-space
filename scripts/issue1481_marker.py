#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002, RUF003  # marker glyph + em-dash intentional
"""#1481 marker A4/A5 dispatcher — 48-cell marker half of the con-vs-pos grid.

Grid (plan §4.1 A4/A5): behavior = marker ` ※` (token id 83399); context ∈
{persona software_engineer, bare default, WildChat 2-turn real prefix, 2-shot
marker ICL prefix}; regime ∈ {con, po}; lr ∈ {5e-6, 1e-5, 1e-4}; seed ∈
{42, 137}. Dispatch groups: ``marker-a`` = pers+bare (24 cells), ``marker-b``
= conv+icl (24 cells) — launched by ``issue1481_dispatch.sh``.

Composition: the #1112/#1333 marker rig verbatim except the LR ladder and the
factory negative panel — cfg-independent primitives are IMPORTED from
``issue1333_dispatch`` (engine lifecycle, greedy chunking, slot reads, ICL
demo construction, rollout persistence) and ``experiments.issue_1333``
(mix row shape, rendered-token disjointness, question sampling); the cell
registry, LR ladder, factory-panel mixes, per-rung ladder, [5, 12]-nat
selection, and 6-context panel battery live here.

Phases (checkpoint-per-phase, resume-keyed):
  mixes   ICL bank fill + greedy base R maps (ONE engine) + 4 con / 4 po
          factory-panel mixes (token-budget gated, sha-pinned, uploaded
          BEFORE training)
  cells   per-cell units over a work-conserving 1-GPU fanout: train (LoRA
          ladder, band-stop LOG-ONLY) -> rung upload -> P3 gauge assert ->
          per-rung slot-read ladder (fresh greedy 20-q gens + teacher-forced
          four-float reads, trained AND base) -> P2 apply-path gate ->
          [5, 12]-nat earliest-rung selection with panel de-saturation
          confirm -> 6-context panel battery at selected/onset/ceiling rungs
          -> JSON/rollout uploads
  summary con-vs-po dose-match table + sentinel

``--smoke`` is the SAME dispatcher at tiny knobs (tiny-real: real tokenizer /
collator / band callback / PEFT round-trip on the from-config tiny Qwen2;
seams replace ONLY vLLM generation + the Hub boundary), running BOTH regimes
and both context classes (persona/bare + prefix) through the same unit path,
including the subprocess fanout.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

# vLLM v1 EngineCore fork-poisoning guard (gotchas.md #628): set BEFORE any
# vllm import — this dispatcher touches tokenizers/transformers pre-LLM().
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1090_fu3_cells as fu3_cells  # noqa: E402
import issue1112_dispatch as d1112  # noqa: E402

# Reused #1333 primitives (cfg-independent, or duck-typed on ``cfg.out_root``):
# engine lifecycle (_vllm_engine/_reap_engine/_wait_engine_release), chunked
# greedy, strip-at-marker, tokenizer + P0 assert, ICL demo construction,
# rollout persistence, trajectory parsing, fanout unit reaping, GPU hygiene.
import issue1333_dispatch as d1333  # noqa: E402

from explore_persona_space.artifacts import negatives as neg_mod  # noqa: E402
from explore_persona_space.artifacts.context import (  # noqa: E402
    CONTEXTS,
    Context,
    icl_prefix_context,
)
from explore_persona_space.artifacts.organisms import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    enforce_mix_token_budget,
)
from explore_persona_space.artifacts.recipe import (  # noqa: E402
    build_train_config,
    recipe_for,
)
from explore_persona_space.experiments import issue_1333 as C  # noqa: E402
from explore_persona_space.experiments.factor_screen_365.persona_panel import (  # noqa: E402
    EVAL_QUESTIONS_20,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1481.marker")

ISSUE = 1481
# == issue1481_cells.DATA_PREFIX_1481 (kept import-light — issue1481_cells
# pulls the whole fu4/1434 registry chain at import; string pinned here).
DATA_PREFIX = "issue1481_conpos_grid/marker"
ADAPTER_PREFIX = "issue1481/marker"  # model-repo run dirs: issue1481/marker/<run_id>/
HF_DATA_REPO = C.HF_DATA_REPO
HF_MODEL_REPO = "superkaiba1/explore-persona-space"

# ── Banks (both sha-pinned in-repo; fixes #1333's 10/20 train–eval overlap) ──
QUERY_BANKS_DIR = REPO_ROOT / "src/explore_persona_space/artifacts/query_banks"
TRAIN_BANK_FILE = "issue1481_marker_train10_v1.json"
TRAIN_BANK_SHA256 = "01b79a29c1aaa670a5cd76934796b8ab0b1f5a091fe34423a38e9c4e8d74e94a"
# Canonical-JSON sha of the 20-question marker eval bank (EVAL_QUESTIONS_20).
EVAL_BANK_SHA256 = "7c08c15bea17f750d0c74f6e3d484644e4c3e570f157a9686653f8c0b12f6c46"

# ── Selection / gates (plan §4.3/§5) ─────────────────────────────────────────
INSTALL_WINDOW = (5.0, 12.0)  # nats, ΔG gate
DOSE_MATCH_TOL_NATS = 1.5  # |ΔG_con − ΔG_po| for a dose-MATCHED pair
ARGMAX_CEILING = 0.92  # #1333 parity: per-context argmax saturation ceiling
SAFETY_BAND = (25.0, 30.0)  # in-loop band, LOG-ONLY (full ladders are the deliverable)
APPLY_GATE_HALT_NATS = 2.0  # P2: in-loop rose >= this ...
APPLY_GATE_FLAT_NATS = 0.5  # ... while the off-line ladder stayed flat under this
MIX_SEED = 42  # mixes are seed-INVARIANT (one mix per context × regime)
POS_EX = 200
NEG_TOTAL = 800  # #1112/#1333 marker-line 200:800 convention
MIX_MAX_LENGTH = 2048  # trainer max_length — the token-budget gate's budget

# ── Cell grid ────────────────────────────────────────────────────────────────
CTX_KEYS = ("pers", "bare", "conv", "icl")
REGIMES = ("con", "po")
SEEDS = (42, 137)
# lr_key -> (lr, ceiling steps, base save cadence) — plan §4.2 LR ladder table.
LR_ARMS: dict[str, tuple[float, int, int]] = {
    "lr5e6": (5e-6, 400, 10),
    "lr1e5": (1e-5, 200, 10),
    "lr1e4": (1e-4, 100, 5),
}
PO_DENSE_SAVE_STEPS = 5  # po arms at lr5e6/lr1e5: every 5 steps for steps <= 60
PO_DENSE_UNTIL = 60
GROUPS = {"marker-a": ("pers", "bare"), "marker-b": ("conv", "icl")}

CTX_SOURCE_ID = {
    "pers": "persona_software_engineer",
    "bare": "default",
    "conv": fu3_cells.CONV_CONTEXT_ID,  # wildchat_prefix_real545
    "icl": "icl_prefix_marker",
}
# identity keys for the panel-∩-sources disjointness assert (#527/#538)
CTX_SOURCE_IDENTITY = {
    "pers": "software_engineer",
    "bare": "default",
    "conv": fu3_cells.CONV_CONTEXT_ID,
    "icl": "icl_prefix_marker",
}


@dataclasses.dataclass(frozen=True)
class CellSpec:
    run_id: str
    ctx_key: str
    regime: str
    lr_key: str
    seed: int

    @property
    def lr(self) -> float:
        return LR_ARMS[self.lr_key][0]

    @property
    def ceiling(self) -> int:
        return LR_ARMS[self.lr_key][1]

    @property
    def train_save_steps(self) -> int:
        """TRAINING cadence — the densest cadence any deliverable rung needs;
        off-cadence rungs above PO_DENSE_UNTIL are pruned post-train."""
        if self.lr_key == "lr1e4":
            return 5
        if self.regime == "po":
            return PO_DENSE_SAVE_STEPS
        return LR_ARMS[self.lr_key][2]

    def keep_rung(self, step: int) -> bool:
        """Deliverable rung cadence (plan §4.2 table): lr1e4 every 5; po arms
        at lr5e6/1e-5 every 5 up to step 60 then every 10; con arms every 10."""
        if self.lr_key == "lr1e4":
            return step % 5 == 0
        if self.regime == "po":
            return step % (PO_DENSE_SAVE_STEPS if step <= PO_DENSE_UNTIL else 10) == 0
        return step % 10 == 0


def run_id_for(ctx_key: str, regime: str, lr_key: str, seed: int) -> str:
    return f"mk-{ctx_key}-{regime}-{lr_key}-s{seed}"


def parse_cell(run_id: str) -> CellSpec:
    parts = run_id.split("-")
    if len(parts) != 5 or parts[0] != "mk" or not parts[4].startswith("s"):
        raise ValueError(f"bad marker run_id {run_id!r}: want mk-<ctx>-<regime>-<lr>-s<seed>")
    _, ctx_key, regime, lr_key, seed_tok = parts
    seed = int(seed_tok[1:])
    if ctx_key not in CTX_KEYS or regime not in REGIMES or lr_key not in LR_ARMS:
        raise ValueError(f"bad marker run_id {run_id!r}: unknown component")
    if seed not in SEEDS:
        raise ValueError(f"bad marker run_id {run_id!r}: seed {seed} not in {SEEDS}")
    return CellSpec(run_id, ctx_key, regime, lr_key, seed)


ALL_CELLS = tuple(
    run_id_for(c, r, k, s) for c in CTX_KEYS for r in REGIMES for k in LR_ARMS for s in SEEDS
)
# Both regimes + one persona/bare and one prefix-class context + two LR arm
# classes ride the smoke (per-arm-class duty).
SMOKE_CELLS = ("mk-pers-con-lr5e6-s42", "mk-icl-po-lr1e4-s137")


# ── Banks ────────────────────────────────────────────────────────────────────


def train_questions() -> list[str]:
    """The sha-pinned 10-question train bank, asserted DISJOINT from the
    20-question eval bank at every load (plan §4.2)."""
    path = QUERY_BANKS_DIR / TRAIN_BANK_FILE
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != TRAIN_BANK_SHA256:
        raise RuntimeError(f"train bank sha drift: {sha} != {TRAIN_BANK_SHA256} at {path}")
    qs = json.loads(raw)
    if not (isinstance(qs, list) and len(qs) == 10 and len(set(qs)) == 10):
        raise RuntimeError(f"train bank malformed: want 10 unique questions, got {len(qs)}")
    overlap = set(qs) & set(EVAL_QUESTIONS_20)
    if overlap:
        raise RuntimeError(f"train bank overlaps the eval bank: {len(overlap)} question(s)")
    return list(qs)


def eval_questions(cfg: Cfg) -> list[str]:
    """The sha-pinned 20-question marker eval bank (EVAL_QUESTIONS_20)."""
    sha = hashlib.sha256(
        json.dumps(list(EVAL_QUESTIONS_20), ensure_ascii=False).encode()
    ).hexdigest()
    if sha != EVAL_BANK_SHA256:
        raise RuntimeError(f"eval bank sha drift: {sha} != {EVAL_BANK_SHA256}")
    qs = list(EVAL_QUESTIONS_20)
    if cfg.eval_question_limit is not None:
        qs = qs[: cfg.eval_question_limit]
    return qs


# ── Config + seams ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    smoke: bool
    cells: tuple[str, ...]
    out_root: Path
    eval_question_limit: int | None = None
    upload: bool = True
    sentinel_dir: Path | None = None
    phases: tuple[str, ...] = ()  # empty -> all
    group: str | None = None

    def regime_key(self, run_id: str) -> dict:
        spec = parse_cell(run_id)
        return {
            "issue": ISSUE,
            "smoke": self.smoke,
            "run_id": run_id,
            "lr": spec.lr,
            "max_steps": spec.ceiling,
            "train_save_steps": spec.train_save_steps,
            "eval_question_limit": self.eval_question_limit,
            "window": list(INSTALL_WINDOW),
            "banks": {"train": TRAIN_BANK_SHA256, "eval": EVAL_BANK_SHA256},
            "marker": [C.MARKER_TEXT, C.MARKER_TOKEN_ID],
        }


@dataclasses.dataclass
class Seams:
    """Injectable boundaries; every field ``None``/default -> the real path.
    ``--smoke`` populates: the vLLM generation boundary (deterministic stub),
    a compute-scale train clamp (tiny-real), a recording upload fn, and
    device=cpu for the REAL teacher-forced slot-read bodies."""

    gen_backend_factory: Callable[..., object] | None = None
    train_clamp: Callable[[object], object] | None = None
    upload_fn: Callable[..., str] | None = None
    device: str = "cuda:0"


class VllmBackend:
    """ONE enable_lora engine per unit; per-rung LoRARequest (the #1090
    shared-engine pattern — no per-rung merges)."""

    def __init__(self, model_path: str = DEFAULT_BASE_MODEL, *, enable_lora: bool = False):
        self._llm = d1333._vllm_engine(model_path, enable_lora=enable_lora)
        self._seq = 0

    def generate(self, prompts: list[str], max_new: int, *, adapter_dir=None) -> list[str]:
        req = None
        if adapter_dir is not None:
            from vllm.lora.request import LoRARequest

            self._seq += 1
            req = LoRARequest(f"i1481_{self._seq}", self._seq, str(adapter_dir))
        return d1333._greedy(self._llm, prompts, max_new, lora_request=req)

    def close(self, label: str) -> None:
        d1333._reap_engine(self._llm)
        d1333._wait_engine_release(label=label)


class SmokeBackend:
    """Deterministic canned generations (the vLLM boundary is the ONLY faked
    model surface — slot reads run the real bodies on the tiny Qwen2)."""

    def generate(self, prompts: list[str], max_new: int, *, adapter_dir=None) -> list[str]:
        del max_new
        out = []
        for i, p in enumerate(prompts):
            text = f"Smoke answer {hashlib.sha1(p.encode()).hexdigest()[:8]} probe {i}."
            if adapter_dir is not None and i % 2 == 0:
                # trained side: alternate emissions exercise strip-at-marker +
                # the emission-rate aggregates
                text = f"{text}{C.MARKER_SEP}※ trailing"
            out.append(text)
        return out

    def close(self, label: str) -> None:
        del label


def make_smoke_seams(cfg: Cfg) -> Seams:
    """Tiny-real seams (the #906/#1090 pattern): 7B weights -> from-config tiny
    Qwen2 over the REAL vocab; real tokenizer / collator / band callback /
    PEFT round-trip; vLLM + Hub boundaries stubbed."""
    import issue1090_run as run1090

    run1090._install_tiny_qwen(42)

    def train_clamp(train_cfg):
        return dataclasses.replace(
            train_cfg,
            max_steps=2,
            save_steps=1,
            batch_size=1,
            grad_accum=1,
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
            gradient_checkpointing=False,
            bf16=False,  # TrainingArguments rejects bf16 on CPU-only machines
            logging_steps=1,
            report_to="none",  # WANDB_INTENTIONALLY_DISABLED: offline CPU smoke
            # dense in-loop probes: >=1 trajectory point within the 2-step train
            marker_band_eval_every_steps=1,
            marker_band_dense_until=2,
            marker_band_min_steps=2,
            marker_band_probe_max_rows=4,
        )

    def upload_fn(local, repo_id, repo_type, path_in_repo, **kwargs) -> str:
        rec = {
            "local": str(local),
            "repo_id": repo_id,
            "repo_type": repo_type,
            "path_in_repo": path_in_repo,
        }
        log_path = cfg.out_root / "upload_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return f"smoke://{path_in_repo}"

    return Seams(
        gen_backend_factory=lambda *a, **k: SmokeBackend(),
        train_clamp=train_clamp,
        upload_fn=upload_fn,
        device="cpu",
    )


def make_seams(cfg: Cfg) -> Seams:
    return make_smoke_seams(cfg) if cfg.smoke else Seams()


def _backend(seams: Seams, *, enable_lora: bool = False):
    if seams.gen_backend_factory is not None:
        return seams.gen_backend_factory(enable_lora=enable_lora)
    return VllmBackend(enable_lora=enable_lora)


def _upload(
    cfg: Cfg,
    seams: Seams,
    local: Path,
    path_in_repo: str,
    *,
    repo_id: str = HF_DATA_REPO,
    repo_type: str = "dataset",
    as_file: bool = False,
) -> str:
    if not cfg.upload:
        return "skipped://no-upload"
    if seams.upload_fn is not None:
        return seams.upload_fn(local, repo_id, repo_type, path_in_repo)
    url = hub._upload(local, repo_id, repo_type, path_in_repo, upload_as_file=as_file)
    if not str(url):
        raise RuntimeError(f"upload of {local} -> {path_in_repo} returned no path")
    return str(url)


# ── Contexts ─────────────────────────────────────────────────────────────────


def _icl_context(cfg: Cfg) -> Context:
    bank_dir = cfg.out_root / "inputs"
    if not (bank_dir / "icl_examples_marker.json").exists():
        raise RuntimeError(
            f"marker ICL bank not filled under {bank_dir} — run the mixes phase first"
        )
    return icl_prefix_context("marker", bank_dir=bank_dir)


def source_context(cfg: Cfg, ctx_key: str) -> Context:
    if ctx_key == "pers":
        return CONTEXTS["persona_software_engineer"]
    if ctx_key == "bare":
        return CONTEXTS["default"]
    if ctx_key == "conv":
        fu3_cells.register_fu3_contexts()
        return CONTEXTS[fu3_cells.CONV_CONTEXT_ID]
    if ctx_key == "icl":
        return _icl_context(cfg)
    raise ValueError(f"unknown ctx_key {ctx_key!r}")


def training_negative_panel(ctx_key: str):
    """The factory negative panel (artifacts.negatives default_v1): 4 members
    + the default assistant; bare-default cells use the 4 non-default members
    (the default assistant is that arm's SOURCE — #527/#538 disjointness)."""
    panel = neg_mod.default_panel()
    if ctx_key == "bare":
        panel = tuple(m for m in panel if m.identity != "default")
    return panel


def panel_contexts(cfg: Cfg) -> dict[str, Context]:
    """The FIXED 6-context marker eval panel (mirrors
    issue1090_fu3_worker.bystander_panel): the 4 grid source contexts + 2
    held-out personas from the factory panel (persona-kind members)."""
    fu3_cells.register_fu3_contexts()
    ordered = [
        CONTEXTS["persona_software_engineer"],
        CONTEXTS["default"],
        CONTEXTS[fu3_cells.CONV_CONTEXT_ID],
        _icl_context(cfg),
    ]
    panel = {c.context_id: c for c in ordered}
    held = 0
    for member in neg_mod.default_panel():
        c = member.to_context()
        if c.context_id in panel or c.kind != "persona":
            continue
        panel[c.context_id] = c
        held += 1
        if held == 2:
            break
    if len(panel) != 6:
        raise RuntimeError(f"marker panel has {len(panel)} contexts, want 6: {sorted(panel)}")
    return panel


# ── Mixes phase ──────────────────────────────────────────────────────────────


def _mix_path(cfg: Cfg, ctx_key: str, regime: str) -> Path:
    return cfg.out_root / "mixes" / f"marker_{ctx_key}_{regime}.jsonl"


def _fill_icl_bank(cfg: Cfg, tok, backend) -> Path:
    """Fill the committed questions-only marker ICL spec with greedy base
    answers + marker (the #1333 construction), collision-checked against BOTH
    #1481 banks."""
    dest = cfg.out_root / "inputs" / "icl_examples_marker.json"
    if dest.exists():
        return dest
    spec = json.loads((QUERY_BANKS_DIR / C.ICL_QUESTIONS_SPEC).read_text())
    demo_qs = [ex["question"] for ex in spec["examples"]]
    if len(demo_qs) != 2:
        raise RuntimeError(f"marker ICL spec must hold 2 demo questions, got {len(demo_qs)}")
    banned = set(train_questions()) | set(EVAL_QUESTIONS_20)
    for q in demo_qs:
        if q in banned:
            raise RuntimeError("ICL demo question collides with the train/eval banks")
    bare = CONTEXTS["default"]
    answers = backend.generate([bare.render(tok, q) for q in demo_qs], C.R_GEN_MAX_NEW_TOKENS)
    examples = []
    for q, a in zip(demo_qs, answers, strict=True):
        if "※" in a:
            raise RuntimeError("greedy demo answer already carries the marker")
        examples.append({"question": q, "answer": d1333._icl_demo_answer(a)})
    d1333._assert_icl_demo_tails_encode_marker(tok, examples)
    bank = {**spec, "examples": examples, "filled_by": "issue1481_marker phase_mixes"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n")
    d1333._persist_rollouts(cfg, "mixes", "icl_bank", bank)
    return dest


def _greedy_r_map(
    cfg: Cfg, tok, backend, msgs_for_q, questions: list[str], label: str
) -> dict[str, str]:
    """Greedy base R per question under one context (cached per label;
    truncation rate logged — #1333 parity)."""
    cache = cfg.out_root / "mixes" / f"r_{label}.json"
    if cache.exists():
        return d1333._read_json(cache)["r_by_q"]
    prompts = [
        tok.apply_chat_template(msgs_for_q(q), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    responses = backend.generate(prompts, C.R_GEN_MAX_NEW_TOKENS)
    n_trunc = sum(
        1
        for r in responses
        if len(tok.encode(r, add_special_tokens=False)) >= C.R_GEN_MAX_NEW_TOKENS
    )
    logger.info("[mixes] %s: %d/%d generations at the cap", label, n_trunc, len(responses))
    rec = {
        "label": label,
        "truncation_rate": n_trunc / len(responses),
        "r_by_q": dict(zip(questions, responses, strict=True)),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    d1333._atomic_json(cache, rec)
    d1333._persist_rollouts(cfg, "mixes", label, rec)
    return rec["r_by_q"]


def _build_ctx_mixes(cfg: Cfg, tok, ctx_key: str, r_src, r_neg_by_slug) -> dict[str, dict]:
    """con + po mixes for one context: 200 positives (10 train q × 20 repeats,
    greedy base R + ` ※` at the end slot) + 800 factory-panel negatives; po =
    the positive partition. Asserts the exact 200/800 pre-budget split; the
    plan-mandated token-budget gate (pair-drop, fail-loud > 0.10) then runs."""
    src = source_context(cfg, ctx_key)
    panel = training_negative_panel(ctx_key)
    quota, rem = divmod(NEG_TOTAL, len(panel))
    if rem:
        raise RuntimeError(f"panel size {len(panel)} does not divide {NEG_TOTAL}")
    qs = train_questions()

    pos_rows = []
    for q in C.sample_questions(qs, POS_EX, random.Random(MIX_SEED)):
        r = r_src[q]
        if "※" in r:
            raise RuntimeError(f"positive R under {ctx_key} carries the marker")
        pos_rows.append(C.make_row(src.messages(q), f"{r}{C.MARKER_SEP}{C.MARKER_TEXT}"))
    cn_rows = []
    for j, member in enumerate(panel):
        rng_j = random.Random(MIX_SEED + 1000 + j)
        for q in C.sample_questions(qs, quota, rng_j):
            r = r_neg_by_slug[member.slug][q]
            if "※" in r:
                raise RuntimeError(f"negative R for {member.slug} carries the marker")
            cn_rows.append(C.make_row(member.messages(q), r))
    if (len(pos_rows), len(cn_rows)) != (POS_EX, NEG_TOTAL):
        raise RuntimeError(
            f"marker mix {ctx_key}: {len(pos_rows)}/{len(cn_rows)} != {POS_EX}/{NEG_TOTAL}"
        )
    C.assert_positive_tails_encode_marker(tok, pos_rows)
    kept_pos, kept_cn, _, budget_stats = enforce_mix_token_budget(
        pos_rows, cn_rows, tok, MIX_MAX_LENGTH, label=f"mk-{ctx_key}", log=logger
    )

    manifests: dict[str, dict] = {}
    for regime, rows in (("con", kept_pos + kept_cn), ("po", list(kept_pos))):
        out = _mix_path(cfg, ctx_key, regime)
        shuffled = list(rows)
        random.Random(MIX_SEED).shuffle(shuffled)
        sha = C.write_mix(shuffled, out)
        manifest = {
            "ctx_key": ctx_key,
            "regime": regime,
            "source_context": CTX_SOURCE_ID[ctx_key],
            "negatives": [m.slug for m in panel] if regime == "con" else [],
            "n_total": len(shuffled),
            "n_positive": len(kept_pos),
            "n_negative": len(kept_cn) if regime == "con" else 0,
            "prebudget_split": [POS_EX, NEG_TOTAL],
            "token_budget": budget_stats,
            "marker_text": C.MARKER_TEXT,
            "mix_seed": MIX_SEED,
            "sha256": sha,
            "train_bank_sha256": TRAIN_BANK_SHA256,
            "r_provenance": "fresh greedy base R (max_new_tokens 1024), positives + negatives",
        }
        out.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        manifests[regime] = manifest
    return manifests


def phase_mixes(cfg: Cfg, seams: Seams) -> dict:
    d1333._phase("i1481_marker_mixes")
    tok = d1333._tokenizer()  # P0 token-id assert (HALT) lives inside
    needed_ctx = sorted({parse_cell(c).ctx_key for c in cfg.cells}, key=CTX_KEYS.index)
    manifests: dict[str, dict] = {}
    backend = _backend(seams, enable_lora=False)
    try:
        _fill_icl_bank(cfg, tok, backend)

        # Disjointness invariants per context arm (panel ∩ realized sources = ∅,
        # identity level AND rendered-token level) — BEFORE any mix writes.
        for ctx_key in needed_ctx:
            panel = training_negative_panel(ctx_key)
            src = source_context(cfg, ctx_key)
            neg_mod.assert_panel_disjoint_from_sources(
                panel,
                [src.context_id],
                source_identities={src.context_id: CTX_SOURCE_IDENTITY[ctx_key]},
            )
            C.assert_rendered_disjoint(
                tok,
                source_id=src.context_id,
                source_msgs_for_q=src.messages,
                panel={m.slug: m.messages for m in panel},
                questions=train_questions() if not cfg.smoke else train_questions()[:2],
            )

        qs = train_questions()
        member_by_slug = {m.slug: m for m in neg_mod.default_panel()}
        needed_slugs = sorted(
            {m.slug for ctx_key in needed_ctx for m in training_negative_panel(ctx_key)}
        )
        r_neg_by_slug = {
            slug: _greedy_r_map(cfg, tok, backend, member_by_slug[slug].messages, qs, f"neg_{slug}")
            for slug in needed_slugs
        }
        for ctx_key in needed_ctx:
            con_path = _mix_path(cfg, ctx_key, "con")
            if con_path.exists() and _mix_path(cfg, ctx_key, "po").exists():
                for regime in REGIMES:
                    man_path = _mix_path(cfg, ctx_key, regime).with_suffix(".manifest.json")
                    manifests[f"{ctx_key}_{regime}"] = d1333._read_json(man_path)
                continue
            r_src = _greedy_r_map(
                cfg, tok, backend, source_context(cfg, ctx_key).messages, qs, f"src_{ctx_key}"
            )
            built = _build_ctx_mixes(cfg, tok, ctx_key, r_src, r_neg_by_slug)
            for regime, man in built.items():
                manifests[f"{ctx_key}_{regime}"] = man
    finally:
        backend.close("i1481-mixes")

    # Upload mixes + manifests + the filled ICL bank BEFORE training.
    for ctx_key in needed_ctx:
        for regime in REGIMES:
            p = _mix_path(cfg, ctx_key, regime)
            _upload(cfg, seams, p, f"{DATA_PREFIX}/mixes/{p.name}", as_file=True)
            man = p.with_suffix(".manifest.json")
            _upload(cfg, seams, man, f"{DATA_PREFIX}/mixes/{man.name}", as_file=True)
    bank = cfg.out_root / "inputs" / "icl_examples_marker.json"
    _upload(cfg, seams, bank, f"{DATA_PREFIX}/inputs/{bank.name}", as_file=True)
    return manifests


# ── Training (per-cell unit, part 1) ─────────────────────────────────────────


def marker_lora_config_1481(spec: CellSpec, *, tokenizer, cell_root: Path):
    """The #1112/#1333 marker recipe verbatim except the LR ladder + LOG-ONLY
    band (plan §4.2): recipe_for('marker') -> lr override at SPEC level, then
    ceiling/save-cadence/warmup at config level (the #1333 pattern).
    save_total_limit=None — the #641 pruning trap would delete ladder rungs."""
    train_recipe = recipe_for("marker", arm="primary")
    train_recipe = dataclasses.replace(
        train_recipe,
        overrides={
            **train_recipe.overrides,
            "lr": spec.lr,
            "marker_band_low_nats": SAFETY_BAND[0],
            "marker_band_high_nats": SAFETY_BAND[1],
        },
    )
    tcfg = build_train_config(
        train_recipe,
        run_name=f"issue1481_{spec.run_id}",
        seed=spec.seed,
        tokenizer=tokenizer,
    )
    return dataclasses.replace(
        tcfg,
        max_steps=spec.ceiling,
        save_steps=spec.train_save_steps,
        save_total_limit=None,
        warmup_steps=C.LORA_WARMUP_STEPS,  # 5 — the m1 warmup-swamp fix
        marker_band_log_only=True,  # NEVER stops — full ladders are the deliverable
        marker_band_trajectory_path=str(cell_root / "band_trajectory.json"),
    )


def _prune_off_cadence_rungs(spec: CellSpec, train_dir: Path) -> list[int]:
    """Delete trained-at-dense-cadence rungs the plan's deliverable table does
    not keep (po arms above step 60 revert to the 10-step cadence)."""
    pruned = []
    for step, path in sorted(d1112._enumerate_rungs(train_dir).items()):
        if not spec.keep_rung(step) and step != spec.ceiling:
            shutil.rmtree(path, ignore_errors=True)
            pruned.append(step)
    if pruned:
        logger.info("[train] %s: pruned %d off-cadence rungs", spec.run_id, len(pruned))
    return pruned


def _assert_rslora_gauge(spec: CellSpec, train_dir: Path) -> dict:
    """P3 (HALT): the trained adapter's OWN adapter_config.json matches the arm
    spec — r16/α32, rsLoRA, attention-only targets, no modules_to_save."""
    rungs = d1112._enumerate_rungs(train_dir)
    final = rungs[max(rungs)]
    cfg_json = json.loads((final / "adapter_config.json").read_text())
    problems = []
    if cfg_json.get("r") != 16 or cfg_json.get("lora_alpha") != 32:
        problems.append(f"r/alpha {cfg_json.get('r')}/{cfg_json.get('lora_alpha')} != 16/32")
    if cfg_json.get("use_rslora") is not True:
        problems.append(f"use_rslora {cfg_json.get('use_rslora')!r} != True")
    targets = set(cfg_json.get("target_modules") or ())
    if targets != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        problems.append(f"target_modules {sorted(targets)}")
    if targets & {"lm_head", "embed_tokens"}:
        problems.append("target_modules include lm_head/embed_tokens")
    if cfg_json.get("modules_to_save"):
        problems.append(f"modules_to_save {cfg_json['modules_to_save']!r} non-empty")
    if problems:
        raise RuntimeError(f"[P3-gauge] {spec.run_id}: {'; '.join(problems)} ({final})")
    return {"verdict": "pass", "checkpoint": str(final)}


def _train_cell(cfg: Cfg, seams: Seams, spec: CellSpec) -> dict:
    from explore_persona_space.train.sft import train_lora

    cell_root = cfg.out_root / spec.run_id
    done = cell_root / "build_result.json"
    if done.exists():
        return d1333._read_json(done)
    tok = d1333._tokenizer()
    mix = _mix_path(cfg, spec.ctx_key, spec.regime)
    if not mix.exists():
        raise RuntimeError(f"mix {mix} missing — run the mixes phase first")
    # Ground the sentinel card's wandb_project claim (report_to=wandb runs).
    os.environ.setdefault("WANDB_PROJECT", "issue1481_marker")
    train_cfg = marker_lora_config_1481(spec, tokenizer=tok, cell_root=cell_root)
    if seams.train_clamp is not None:
        train_cfg = seams.train_clamp(train_cfg)
    adapter_dir, loss = train_lora(
        DEFAULT_BASE_MODEL, str(mix), str(cell_root / "train"), cfg=train_cfg
    )
    import torch

    if torch.cuda.is_available():
        from explore_persona_space.artifacts.organisms import release_trainer_cuda_memory

        release_trainer_cuda_memory()
    train_dir = cell_root / "train"
    pruned = [] if cfg.smoke else _prune_off_cadence_rungs(spec, train_dir)
    gauge = _assert_rslora_gauge(spec, train_dir)
    # Rung adapters upload BEFORE any long consumer (the ladder) — plan §6.5.
    _upload(
        cfg,
        seams,
        train_dir,
        f"{ADAPTER_PREFIX}/{spec.run_id}",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )
    rec = {
        "adapter_root": str(adapter_dir),
        "train_dir": str(train_dir),
        "training_loss": float(loss),
        "rungs": sorted(d1112._enumerate_rungs(train_dir)),
        "pruned_rungs": pruned,
        "gauge": gauge,
        "trajectory": str(cell_root / "band_trajectory.json"),
        "mix_sha256": d1333._read_json(mix.with_suffix(".manifest.json"))["sha256"],
    }
    d1333._atomic_json(done, rec)
    return rec


# ── Slot-read ladder (per-cell unit, part 2) ─────────────────────────────────


def _load_base(device: str):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict = {"token": os.environ.get("HF_TOKEN")}
    if device.startswith("cuda"):
        kwargs.update(torch_dtype=torch.bfloat16, device_map={"": device})
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_BASE_MODEL, **kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model


def _slot_read(model, tok, contexts: list[str], device: str) -> list[dict]:
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    return compute_marker_slot_stats(
        model,
        tok,
        contexts,
        C.MARKER_TEXT,
        device=device,
        eos_token_id=C.EOS_TOKEN_ID,
        include_argmax=True,
    )


def _ladder_record(meta, trained, base, *, text_emitted: list[bool]) -> dict:
    rec = d1333._delta_record(meta, trained, base)
    margins = [
        (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
        for t, b in zip(trained, base, strict=True)
    ]
    rec["delta_margin_mean"] = float(sum(margins) / len(margins))
    rec["gen_emission_rate"] = float(sum(text_emitted) / len(text_emitted))
    return rec


def _ladder_cell(cfg: Cfg, seams: Seams, spec: CellSpec, backend, base_model, tok) -> dict:
    """Per-rung slot-read ladder: fresh greedy 20-q gens (max_new 2048) +
    teacher-forced four-float reads, trained AND base, EVERY persisted rung
    (full ladders are the deliverable). Writes slot_reads_rung<step>.json at
    the cell root (plan §6.5 glob) + the ladder.json aggregate."""
    from peft import PeftModel

    cell_root = cfg.out_root / spec.run_id
    ladder_path = cell_root / "ladder.json"
    ladder: dict[int, dict] = {}
    if ladder_path.exists():
        prior = d1333._read_json(ladder_path)
        if prior.get("regime") != cfg.regime_key(spec.run_id):
            raise RuntimeError(f"ladder regime drift under {ladder_path} — fresh --out-root")
        ladder = {int(k): v for k, v in (prior.get("reads_by_step") or {}).items()}

    def _persist() -> None:
        d1333._atomic_json(
            ladder_path,
            {
                "run_id": spec.run_id,
                "regime": cfg.regime_key(spec.run_id),
                "reads_by_step": {str(k): v for k, v in sorted(ladder.items())},
            },
        )

    rungs = d1112._enumerate_rungs(cell_root / "train")
    src = source_context(cfg, spec.ctx_key)
    prompts = [src.render(tok, q) for q in eval_questions(cfg)]
    for step in sorted(rungs):
        if step in ladder:
            continue
        responses = backend.generate(prompts, C.MARKER_MAX_NEW_TOKENS, adapter_dir=rungs[step])
        d1333._persist_rollouts(
            cfg, "ladder", f"{spec.run_id}_rung{step}", {"responses": responses}
        )
        contexts, meta, text_emitted = [], [], []
        for q_idx, (p, r) in enumerate(zip(prompts, responses, strict=True)):
            stripped, emitted = d1333._strip_at_marker(r)
            contexts.append(p + stripped)
            meta.append({"q": q_idx, "gen_emitted": emitted})
            text_emitted.append(emitted)
        peft_model = PeftModel.from_pretrained(base_model, str(rungs[step]))
        trained = _slot_read(peft_model, tok, contexts, seams.device)
        peft_model.unload()
        peft_model = None
        base_stats = _slot_read(base_model, tok, contexts, seams.device)
        rec = _ladder_record(meta, trained, base_stats, text_emitted=text_emitted)
        rec["run_id"], rec["step"] = spec.run_id, step
        d1333._atomic_json(cell_root / f"slot_reads_rung{step}.json", rec)
        ladder[step] = {
            "delta_logp_mean": rec["delta_logp_mean"],
            "delta_margin_mean": rec["delta_margin_mean"],
            "source_emission_rate": rec["source_emission_rate"],
            "gen_emission_rate": rec["gen_emission_rate"],
        }
        _persist()
    _persist()
    return ladder


def _apply_path_gate(cfg: Cfg, spec: CellSpec, ladder: dict[int, dict]) -> dict:
    """P2 (HALT, discriminating): off-line ladder flat at ΔG ≈ 0 across ALL
    rungs while the in-loop trajectory rose ⇒ silent-apply no-op (#534/#1333
    discriminator). The known ~2× in-loop-vs-off-line surface gap is REPORTED,
    never gated."""
    traj_path = cfg.out_root / spec.run_id / "band_trajectory.json"
    in_loop = d1333._trajectory_last_delta(traj_path)
    max_offline = max(abs(rec["delta_logp_mean"]) for rec in ladder.values())
    gate = {
        "in_loop_last_delta": float(in_loop),
        "max_offline_abs_delta": float(max_offline),
        "surface_gap_ratio": float(in_loop / max_offline) if max_offline > 0 else None,
        "verdict": "pass",
    }
    if in_loop >= APPLY_GATE_HALT_NATS and max_offline < APPLY_GATE_FLAT_NATS:
        gate["verdict"] = "halt"
        d1333._atomic_json(cfg.out_root / spec.run_id / "apply_gate.json", gate)
        raise RuntimeError(
            f"[P2-apply-gate] {spec.run_id}: in-loop delta {in_loop:.2f} nats rose while the "
            f"off-line ladder stayed flat (max |ΔG| {max_offline:.3f}) — silent-apply no-op "
            "(#534/#1333 discriminator)"
        )
    d1333._atomic_json(cfg.out_root / spec.run_id / "apply_gate.json", gate)
    return gate


# ── Selection + 6-context panel battery (per-cell unit, part 3) ─────────────


def select_rung_1481(ladder: dict[int, dict], *, window=INSTALL_WINDOW) -> dict:
    """Registered selection (plan §4.3): EARLIEST rung with ΔG ∈ window AND
    source argmax-emission 0/20 AND panel not saturated. No in-window rung ⇒
    the closest-approach-to-window rung, labeled (never a crash — the lr 1e-4
    arm fails de-saturation BY DESIGN and carries the emission map)."""
    if not ladder:
        raise ValueError("empty ladder")
    lo, hi = window
    eligible = [
        s
        for s, rec in ladder.items()
        if lo <= rec["delta_logp_mean"] <= hi
        and rec["source_emission_rate"] == 0.0
        and not rec.get("panel_saturated", False)
    ]
    if eligible:
        step = min(eligible)
        return {
            "step": step,
            "in_window": True,
            "fallback": None,
            "delta_logp_mean": ladder[step]["delta_logp_mean"],
            "window": list(window),
        }

    def _dist(step: int) -> tuple[float, int]:
        dg = ladder[step]["delta_logp_mean"]
        return (max(lo - dg, dg - hi, 0.0), step)

    step = min(ladder, key=_dist)
    return {
        "step": step,
        "in_window": False,
        "fallback": "closest_approach",
        "delta_logp_mean": ladder[step]["delta_logp_mean"],
        "window": list(window),
    }


def _panel_battery(
    cfg: Cfg, seams: Seams, spec: CellSpec, backend, base_model, tok, step: int, roles: list[str]
) -> dict:
    """6-context on-policy battery at one rung: install + leakage + free
    emission (argmax-at-slot AND text-level ※-in-own-generation, per context)
    + the de-saturation fields the selection loop reads."""
    from peft import PeftModel

    cell_root = cfg.out_root / spec.run_id
    out_path = cell_root / "panel" / f"rung{step}.json"
    if out_path.exists():
        rec = d1333._read_json(out_path)
        if set(roles) - set(rec.get("roles", [])):
            rec["roles"] = sorted(set(rec.get("roles", [])) | set(roles))
            d1333._atomic_json(out_path, rec)
        return rec
    panel = panel_contexts(cfg)
    qs = eval_questions(cfg)
    prompts, meta = [], []
    for label, ctx in panel.items():
        for q_idx, q in enumerate(qs):
            prompts.append(ctx.render(tok, q))
            meta.append({"context_id": label, "q": q_idx})
    rung_dir = d1112._enumerate_rungs(cell_root / "train")[step]
    responses = backend.generate(prompts, C.MARKER_MAX_NEW_TOKENS, adapter_dir=rung_dir)
    d1333._persist_rollouts(
        cfg, "panel", f"{spec.run_id}_rung{step}", {"meta": meta, "responses": responses}
    )
    contexts, text_emitted = [], []
    for p, r in zip(prompts, responses, strict=True):
        stripped, emitted = d1333._strip_at_marker(r)
        contexts.append(p + stripped)
        text_emitted.append(emitted)
    peft_model = PeftModel.from_pretrained(base_model, str(rung_dir))
    trained = _slot_read(peft_model, tok, contexts, seams.device)
    peft_model.unload()
    peft_model = None
    base_stats = _slot_read(base_model, tok, contexts, seams.device)

    rec = _ladder_record(meta, trained, base_stats, text_emitted=text_emitted)
    by_ctx: dict[str, dict[str, list[float]]] = {}
    for m, t, b, emitted in zip(meta, trained, base_stats, text_emitted, strict=True):
        d = by_ctx.setdefault(
            m["context_id"], {"deltas": [], "margins": [], "emit": [], "text": []}
        )
        d["deltas"].append(t["logp"] - b["logp"])
        d["margins"].append((t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"]))
        d["emit"].append(1.0 if t.get("argmax_id") == C.MARKER_TOKEN_ID else 0.0)
        d["text"].append(1.0 if emitted else 0.0)
    src_id = CTX_SOURCE_ID[spec.ctx_key]
    rec["per_context"] = {
        k: {
            "delta_logp_mean": float(sum(v["deltas"]) / len(v["deltas"])),
            "delta_margin_mean": float(sum(v["margins"]) / len(v["margins"])),
            "emission_rate": float(sum(v["emit"]) / len(v["emit"])),
            "text_emission_rate": float(sum(v["text"]) / len(v["text"])),
            "is_source": k == src_id,
        }
        for k, v in by_ctx.items()
    }
    rates = {k: v["emission_rate"] for k, v in rec["per_context"].items()}
    rec["saturated"] = any(r >= ARGMAX_CEILING for r in rates.values())
    rec["nonsource_saturated"] = any(r >= ARGMAX_CEILING for k, r in rates.items() if k != src_id)
    rec["run_id"], rec["step"], rec["roles"] = spec.run_id, step, sorted(roles)
    rec["source_context"] = src_id
    d1333._atomic_json(out_path, rec)
    return rec


def _select_cell(cfg: Cfg, seams: Seams, spec: CellSpec, backend, base_model, tok) -> dict:
    """Selection + de-saturation confirm loop + panel batteries at the
    selected / emission-onset / ceiling rungs (plan §4.3/§4.4)."""
    cell_root = cfg.out_root / spec.run_id
    sel_path = cell_root / "selection.json"
    if sel_path.exists():
        return d1333._read_json(sel_path)
    ladder = {
        int(k): v for k, v in d1333._read_json(cell_root / "ladder.json")["reads_by_step"].items()
    }
    rungs = d1112._enumerate_rungs(cell_root / "train")

    tried: set[int] = set()
    while True:
        sel = select_rung_1481(ladder)
        step = sel["step"]
        if not sel["in_window"] or step in tried:
            break
        tried.add(step)
        bys = _panel_battery(cfg, seams, spec, backend, base_model, tok, step, ["candidate"])
        if not bys["saturated"]:
            break
        ladder[step]["panel_saturated"] = True
        d1333._atomic_json(
            cell_root / "ladder.json",
            {
                "run_id": spec.run_id,
                "regime": cfg.regime_key(spec.run_id),
                "reads_by_step": {str(k): v for k, v in sorted(ladder.items())},
            },
        )

    # Emission map (the lr 1e-4 / late-1e-5 deliverable): onset = first rung
    # with any source free emission; ceiling = the last persisted rung.
    onset = next(
        (
            s
            for s in sorted(ladder)
            if ladder[s]["source_emission_rate"] > 0 or ladder[s]["gen_emission_rate"] > 0
        ),
        None,
    )
    ceiling = max(rungs)
    battery_roles: dict[int, list[str]] = {}
    battery_roles.setdefault(int(sel["step"]), []).append("selected")
    if onset is not None:
        battery_roles.setdefault(int(onset), []).append("emission_onset")
    battery_roles.setdefault(int(ceiling), []).append("ceiling")
    batteries = {
        step: _panel_battery(cfg, seams, spec, backend, base_model, tok, step, roles)
        for step, roles in sorted(battery_roles.items())
    }
    # Selectivity break over READ batteries (documented resolution limit):
    # first read rung with any NON-source panel context at the argmax ceiling.
    break_rung = next((s for s in sorted(batteries) if batteries[s]["nonsource_saturated"]), None)
    sel.update(
        {
            "run_id": spec.run_id,
            "ctx_key": spec.ctx_key,
            "regime": spec.regime,
            "lr_key": spec.lr_key,
            "seed": spec.seed,
            "emission_onset_rung": onset,
            "ceiling_rung": int(ceiling),
            "selectivity_break_rung": break_rung,
            "selectivity_break_resolution": sorted(batteries),
            "panel_rungs": {str(s): batteries[s]["roles"] for s in batteries},
        }
    )
    d1333._atomic_json(sel_path, sel)
    return sel


# ── Per-cell unit ────────────────────────────────────────────────────────────


def run_cell_unit(cfg: Cfg, seams: Seams, run_id: str) -> dict:
    """ONE marker cell end-to-end (single GPU; CVD pin authoritative): train →
    rung upload → ladder → P2 gate → selection → panel → JSON uploads."""
    spec = parse_cell(run_id)
    cell_root = cfg.out_root / run_id
    done = cell_root / "cell_result.json"
    if done.exists():
        return d1333._read_json(done)
    tok = d1333._tokenizer()  # P0 token-id assert at the unit entrypoint (HALT)
    build = _train_cell(cfg, seams, spec)

    backend = _backend(seams, enable_lora=True)
    base_model = None
    try:
        base_model = _load_base(seams.device)
        ladder = _ladder_cell(cfg, seams, spec, backend, base_model, tok)
        gate = _apply_path_gate(cfg, spec, ladder)
        sel = _select_cell(cfg, seams, spec, backend, base_model, tok)
    finally:
        if base_model is not None:
            base_model = d1333._free_hf(base_model)
        backend.close(f"i1481-cell[{run_id}]")

    # Cell-terminal uploads: rollout text + every selection/ladder/panel JSON.
    rc_root = cfg.out_root / "raw_completions"
    if rc_root.exists():
        _upload(cfg, seams, rc_root, f"{DATA_PREFIX}/raw_completions")
    _upload(
        cfg, seams, cell_root / "ladder.json", f"{DATA_PREFIX}/{run_id}/ladder.json", as_file=True
    )
    _upload(
        cfg,
        seams,
        cell_root / "selection.json",
        f"{DATA_PREFIX}/{run_id}/selection.json",
        as_file=True,
    )
    for p in sorted(cell_root.glob("slot_reads_rung*.json")):
        _upload(cfg, seams, p, f"{DATA_PREFIX}/{run_id}/{p.name}", as_file=True)
    for p in sorted((cell_root / "panel").glob("*.json")):
        _upload(cfg, seams, p, f"{DATA_PREFIX}/{run_id}/panel/{p.name}", as_file=True)
    traj = cfg.out_root / run_id / "band_trajectory.json"
    if traj.exists():
        _upload(cfg, seams, traj, f"{DATA_PREFIX}/{run_id}/band_trajectory.json", as_file=True)
    _upload(
        cfg,
        seams,
        cell_root / "apply_gate.json",
        f"{DATA_PREFIX}/{run_id}/apply_gate.json",
        as_file=True,
    )

    rec = {
        "run_id": run_id,
        "build": {k: v for k, v in build.items() if k != "per_probe"},
        "apply_gate": gate,
        "selection": {k: v for k, v in sel.items() if k != "per_probe"},
        "n_ladder_rungs": len(ladder),
    }
    d1333._atomic_json(done, rec)
    return rec


# ── Fanout / driver ──────────────────────────────────────────────────────────


def _unit_args(cfg: Cfg, run_id: str) -> list[str]:
    out = [
        "--smoke" if cfg.smoke else "--full",
        "--unit",
        "cell",
        run_id,
        "--cells",
        ",".join(cfg.cells),
        "--out-root",
        str(cfg.out_root),
    ]
    if cfg.eval_question_limit is not None:
        out += ["--eval-question-limit", str(cfg.eval_question_limit)]
    if not cfg.upload:
        out += ["--no-upload"]
    return out


def _fanout_cells(cfg: Cfg, pending: list[str]) -> None:
    """Work-conserving CVD-pinned subprocess pool over self-invocation units
    (1 GPU per cell; the #1333 fanout shape). The smoke rides the SAME
    subprocess path at width 1 with no CVD pin (CPU host — the child
    re-installs its own tiny-real seams from --smoke)."""
    if cfg.smoke:
        ids = ["cpu"]
    else:
        ids = d1333._physical_gpu_ids()
        d1333._gpu_hygiene("i1481-fanout:entry")
    queue = list(pending)
    running: dict[int, tuple[subprocess.Popen, list[str], Path]] = {}
    logs = cfg.out_root / "unit_logs"
    logs.mkdir(parents=True, exist_ok=True)
    while queue or running:
        for g in range(len(ids)):
            if g not in running and queue:
                run_id = queue.pop(0)
                extra = _unit_args(cfg, run_id)
                cmd = ["uv", "run", "python", str(_SCRIPTS_DIR / "issue1481_marker.py"), *extra]
                env = dict(os.environ)
                if not cfg.smoke:
                    cmd += ["--gpu-id", ids[g]]
                    env["CUDA_VISIBLE_DEVICES"] = ids[g]
                log = logs / f"unit_cell_{run_id}_g{g}.log"
                f = open(log, "a")  # noqa: SIM115 — held open for the Popen's lifetime
                running[g] = (
                    subprocess.Popen(
                        cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True
                    ),
                    extra,
                    log,
                )
                logger.info("[fanout] slot %d <- %s (log %s)", g, run_id, log)
        time.sleep(2 if cfg.smoke else 10)
        for g, (proc, extra, log) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[g]
            d1333._reap_completed_unit_group(proc, extra)
            if rc != 0:
                d1112._reap_unit_groups([p for p, _, _ in running.values()])
                logger.error(
                    "[fanout-unit-tail] unit %s rc=%d — last %d lines of %s:\n%s",
                    extra,
                    rc,
                    d1333.SUBPROCESS_TAIL_LINES,
                    log,
                    d1333._tail_lines(log, d1333.SUBPROCESS_TAIL_LINES),
                )
                raise RuntimeError(f"fanout unit {extra} failed rc={rc} (see {logs})")
    if not cfg.smoke:
        d1333._gpu_hygiene("i1481-fanout:exit")


def phase_cells(cfg: Cfg, seams: Seams) -> dict:
    d1333._phase("i1481_marker_cells")
    pending = [c for c in cfg.cells if not (cfg.out_root / c / "cell_result.json").exists()]
    if pending:
        _fanout_cells(cfg, pending)
    out = {}
    for c in cfg.cells:
        out[c] = d1333._read_json(cfg.out_root / c / "cell_result.json")
    return out


def phase_summary(cfg: Cfg, seams: Seams) -> dict:
    """Con-vs-po dose-match table (plan §5): a pair is dose-MATCHED iff both
    regimes' selected ΔG ∈ [5, 12] AND |ΔG_con − ΔG_po| <= 1.5 nats."""
    d1333._phase("i1481_marker_summary")
    pairs = []
    cells = set(cfg.cells)
    for ctx in CTX_KEYS:
        for lrk in LR_ARMS:
            for seed in SEEDS:
                con_id = run_id_for(ctx, "con", lrk, seed)
                po_id = run_id_for(ctx, "po", lrk, seed)
                if con_id not in cells or po_id not in cells:
                    continue
                sels = {}
                for rid in (con_id, po_id):
                    p = cfg.out_root / rid / "selection.json"
                    if not p.exists():
                        raise RuntimeError(f"selection missing for {rid} — cells phase incomplete")
                    sels[rid] = d1333._read_json(p)
                dg_con = sels[con_id]["delta_logp_mean"]
                dg_po = sels[po_id]["delta_logp_mean"]
                both_in = sels[con_id]["in_window"] and sels[po_id]["in_window"]
                pairs.append(
                    {
                        "ctx_key": ctx,
                        "lr_key": lrk,
                        "seed": seed,
                        "con": {"run_id": con_id, "delta_g": dg_con, **_sel_slim(sels[con_id])},
                        "po": {"run_id": po_id, "delta_g": dg_po, **_sel_slim(sels[po_id])},
                        "abs_gap_nats": abs(dg_con - dg_po),
                        "dose_matched": bool(
                            both_in and abs(dg_con - dg_po) <= DOSE_MATCH_TOL_NATS
                        ),
                        "dose_unmatched_flag": not both_in,
                    }
                )
    rec = {
        "issue": ISSUE,
        "window": list(INSTALL_WINDOW),
        "tol_nats": DOSE_MATCH_TOL_NATS,
        "pairs": pairs,
    }
    out = cfg.out_root / "dose_match.json"
    d1333._atomic_json(out, rec)
    _upload(cfg, seams, out, f"{DATA_PREFIX}/dose_match.json", as_file=True)
    return rec


def _sel_slim(sel: dict) -> dict:
    return {
        "step": sel["step"],
        "in_window": sel["in_window"],
        "fallback": sel["fallback"],
        "emission_onset_rung": sel.get("emission_onset_rung"),
        "selectivity_break_rung": sel.get("selectivity_break_rung"),
    }


def write_sentinel(cfg: Cfg, summary: dict) -> Path:
    """End-of-run sentinel in the poll_pipeline envelope (pod-side-reporting.md):
    required keys sentinel_schema_version/kind/version/note; smoke runs write
    kind epm:smoke-result (never epm:results with a smoke flag)."""
    d1333._phase("sentinel")
    sentinel_dir = cfg.sentinel_dir or Path("/workspace/logs")
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    kind = "epm:smoke-result" if cfg.smoke else "epm:results"
    kind_slug = kind.replace(":", "_")
    path = sentinel_dir / f"issue-{ISSUE}-{kind_slug}-{int(time.time())}.json"
    note: dict = {
        "component": "issue1481_marker",
        "group": cfg.group,
        "smoke": cfg.smoke,
        "cells": list(cfg.cells),
        "summary": summary,
    }
    if not cfg.smoke:
        note["reproducibility_card"] = _reproducibility_card(cfg)
    d1333._atomic_json(
        path,
        {
            "sentinel_schema_version": 1,
            "kind": kind,
            "version": 1,
            "task_id": ISSUE,
            "smoke": cfg.smoke,
            "by": "issue1481_marker",
            "ts": int(time.time()),
            "note": json.dumps(note, ensure_ascii=False),
        },
    )
    logger.info("[sentinel] %s", path)
    return path


def _reproducibility_card(cfg: Cfg) -> dict:
    """Per-cell adapter paths (rung dirs upload as issue1481/marker/<run_id>/)
    + WandB fields (entity read from the SDK, never hand-typed)."""
    card: dict = {
        "hf_model_repo": HF_MODEL_REPO,
        "adapter_paths": {c: f"{ADAPTER_PREFIX}/{c}" for c in cfg.cells},
        "wandb_project": os.environ.get("WANDB_PROJECT", "issue1481_marker"),
        "wandb_run_names": [f"issue1481_{c}" for c in cfg.cells],
    }
    try:
        import wandb

        card["wandb_entity"] = wandb.Api().default_entity
    except Exception as e:  # entity resolution is best-effort; verifier probes
        logger.warning("[sentinel] wandb entity resolution failed: %s", e)
    return card


# ── Headroom + CLI ───────────────────────────────────────────────────────────


def _assert_headroom(cfg: Cfg) -> None:
    """Out-root statvfs floor + 1 GB fallocate canary (EDQUOT-aware — the
    d1333 pattern). Floor: ~2 GB/cell of LoRA rungs + rollouts."""
    need_gb = 4.0 if cfg.smoke else max(16.0, 2.5 * len(cfg.cells))
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(cfg.out_root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    if free_gb < need_gb:
        raise RuntimeError(
            f"[disk-headroom] out_root {cfg.out_root} has {free_gb:.1f} GB free < "
            f"required {need_gb:.1f} GB (smoke={cfg.smoke})"
        )
    probe = cfg.out_root / ".headroom_probe"
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        try:
            os.posix_fallocate(fd, 0, 1 << 30)
        finally:
            os.close(fd)
    except OSError as e:
        raise RuntimeError(
            f"[disk-headroom] 1 GB fallocate canary FAILED at {probe} ({e}) — per-pod "
            f"quota (EDQUOT) or wedged filesystem"
        ) from e
    finally:
        probe.unlink(missing_ok=True)


_KNOWN_PHASES = ("mixes", "cells", "summary")


def _default_out_root(smoke: bool) -> Path:
    """Full runs anchor at the backend scratch convention (/workspace/
    eps-issue-1481 — the plan §6.5 deliverable glob root) when /workspace is
    real; local runs at data/issue_1481/marker; smoke at /tmp."""
    if smoke:
        return Path("/tmp/issue-1481-marker-smoke")
    if d1333._repo_on_workspace():
        return Path(f"/workspace/eps-issue-{ISSUE}/out/marker")
    return REPO_ROOT / "data" / f"issue_{ISSUE}" / "marker"


def resolve_cells(args: argparse.Namespace, smoke: bool) -> tuple[str, ...]:
    if args.cells:
        ids = tuple(t.strip() for t in args.cells.split(",") if t.strip())
        for rid in ids:
            parse_cell(rid)  # fail-loud
        return ids
    seeds = (
        tuple(int(t) for t in args.seeds.replace(" ", "").split(",") if t) if args.seeds else SEEDS
    )
    bad = [s for s in seeds if s not in SEEDS]
    if bad:
        raise SystemExit(f"--seeds: unknown seeds {bad} (grid: {SEEDS})")
    if args.group:
        if args.group not in GROUPS:
            raise SystemExit(f"unknown --group {args.group!r}: want one of {sorted(GROUPS)}")
        ctxs = GROUPS[args.group]
        return tuple(
            run_id_for(c, r, k, s) for c in ctxs for r in REGIMES for k in LR_ARMS for s in seeds
        )
    if smoke:
        return SMOKE_CELLS
    raise SystemExit("--full needs --group or --cells")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="#1481 marker A4/A5 phase driver")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="tiny-real, SAME code path")
    mode.add_argument("--full", action="store_true")
    p.add_argument("--group", default=None, help="marker-a (pers+bare) | marker-b (conv+icl)")
    p.add_argument("--cells", default=None, help="comma run_ids (overrides --group)")
    p.add_argument("--seeds", default=None, help="comma seed subset (default 42,137)")
    p.add_argument("--out-root", default=None)
    p.add_argument("--eval-question-limit", type=int, default=None)
    p.add_argument("--no-upload", dest="upload", action="store_false", default=True)
    p.add_argument("--phases", default=None, help="comma subset of mixes,cells,summary")
    p.add_argument("--sentinel-dir", default=None)
    p.add_argument(
        "--unit", nargs=2, metavar=("KIND", "ARG"), default=None, help="internal fanout unit"
    )
    p.add_argument("--gpu-id", type=str, default="0", help="physical GPU (CVD-pinned)")
    return p.parse_args(argv)


def build_cfg(args: argparse.Namespace) -> Cfg:
    smoke = bool(args.smoke)
    out_root = Path(args.out_root) if args.out_root is not None else _default_out_root(smoke)
    phases: tuple[str, ...] = ()
    if args.phases:
        phases = tuple(t.strip() for t in args.phases.split(",") if t.strip())
        bad = [t for t in phases if t not in _KNOWN_PHASES]
        if bad:
            raise SystemExit(f"unknown phases {bad}: want a subset of {_KNOWN_PHASES}")
    return Cfg(
        smoke=smoke,
        cells=resolve_cells(args, smoke),
        out_root=out_root,
        eval_question_limit=(
            args.eval_question_limit
            if args.eval_question_limit is not None
            else (2 if smoke else None)
        ),
        upload=args.upload,
        sentinel_dir=(
            Path(args.sentinel_dir)
            if args.sentinel_dir is not None
            else (out_root / "logs" if smoke and not d1333._repo_on_workspace() else None)
        ),
        phases=phases,
        group=args.group,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = build_cfg(args)
    seams = make_seams(cfg)
    if args.unit is not None:
        kind, arg = args.unit
        if kind != "cell":
            raise ValueError(f"unknown unit kind {kind!r}: want 'cell'")
        if not cfg.smoke:
            d1333._unit_gpu_preflight(kind, args.gpu_id)
        run_cell_unit(cfg, seams, arg)
        return 0
    logger.info(
        "issue1481 marker smoke=%s cells=%d out_root=%s", cfg.smoke, len(cfg.cells), cfg.out_root
    )
    _assert_headroom(cfg)

    def want(phase: str) -> bool:
        return not cfg.phases or phase in cfg.phases

    summary: dict = {"issue": ISSUE, "smoke": cfg.smoke, "n_cells": len(cfg.cells)}
    if want("mixes"):
        summary["mixes"] = {
            k: {"sha256": v["sha256"], "n": v["n_total"]}
            for k, v in phase_mixes(cfg, seams).items()
        }
    if want("cells"):
        cells_out = phase_cells(cfg, seams)
        summary["cells"] = {
            k: {
                "selection": v["selection"].get("step"),
                "in_window": v["selection"].get("in_window"),
            }
            for k, v in cells_out.items()
        }
    if want("summary"):
        dm = phase_summary(cfg, seams)
        summary["dose_matched_pairs"] = sum(1 for p in dm["pairs"] if p["dose_matched"])
        summary["n_pairs"] = len(dm["pairs"])
    write_sentinel(cfg, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
