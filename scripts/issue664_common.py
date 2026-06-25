"""Issue #664 -- shared backbone for the Phase-2 fine-tune fleet (plan v3).

Single source of truth for the #664 grid, context construction, the per-cell
recipes, the contrastive negative panel, and the on-policy elicitation ladder.
The four #664 entry points (build_training_data / dispatch / extract_store /
eval) all import from here so train/eval/extract prompt shapes are identical by
construction.

This is NOT a library module under ``src/``: it lives next to the
``scripts/issue664_*`` entry points (same convention as ``issue594_common.py``
/ ``issue404_common.py``). It is modeled on -- NOT imported from -- the
``origin/issue-537`` reference scripts (``i537_dispatch.py`` /
``i537_build_training_data.py`` / ``i537_contexts.py``), which live only on that
branch. The canonical recipe VALUES are #537's methodology doc §2 + the marker
rules (`.claude/rules/marker-training-recipe.md`).

Design notes carried into the implementation report:

- **Contexts** are the on-``main`` #594 50-context battery
  (``issue594_common.load_battery`` + ``messages_for_instance``), the
  project-canonical context suite. ``i537_contexts.build_messages`` (the model)
  is byte-equivalent to ``messages_for_instance`` for the persona/wildchat/
  rephrase/format/icl/default families; we use the on-``main`` one.
- **EM/insecure-code trains IN-PROCESS via ``train_lora``**, NOT a Hydra
  ``train.py`` subprocess. The plan's §4.4 premise ("``train_lora()`` cannot
  express the linear schedule / warmup_steps / max_steps / adamw_8bit") was
  true for #537's ``train_lora`` but is STALE on ``main``: ``TrainLoraConfig``
  gained ``max_steps`` / ``lr_scheduler_type`` / ``optim`` / ``warmup_steps``
  (the "#545 opt-in overrides", sft.py ~L1424-1434), so the full EM recipe is
  expressible in-process. This UNIFIES every behavior through one code path
  (PASS_UNIFIED smoke) and avoids the missing-config crash (``turner_em.yaml``
  / ``condition=i537_em`` are NOT on ``main``). The named §4.4 divergence.
- **Marker token** ` ※` id 83399 (``i406_conditions.MARKER_ID`` on ``main``),
  asserted at every entrypoint.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("issue664_common")

REPO = Path(__file__).resolve().parents[1]
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"

# ── Marker token (i406_conditions on main; assert at every entrypoint) ────────
from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    MARKER_ID,
    MARKER_TEXT,
)

# Qwen-2.5 <|im_end|> id -- the token the contrastive negatives train at the
# post-response slot (sft.py MarkerOnlyDataCollator default tail). Auto-defaulted
# by train_lora from the tokenizer; pinned here for the marker eval read.
IM_END_ID = 151645

# ── HF / WandB destinations (plan §10 Reproducibility Card) ───────────────────
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_ADAPTER_PREFIX = "adapters/issue_664"
HF_RAW_COMPLETIONS_PREFIX = "issue664_leakage_fleet/raw_completions"
HF_STORE_PREFIX = "theory_assumptions/Qwen2.5-7B-Instruct/issue664"
WANDB_PROJECT = "issue664"

EVAL_ROOT = REPO / "eval_results/issue_664"
DATA_ROOT = REPO / "data/issue_664"
STORE_ROOT = DATA_ROOT / "trained_store"

EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
MAX_NEW_TOKENS = 2048  # >= 2x longest trained completion (CLAUDE.md marker rule)

# ── Battery loader (issue594_common on main; the 50-context activation spine) ──
import issue594_common  # noqa: E402

DEFAULT_SEED = 42
MARKER_REPLICATION_SEED = 1042  # write-direction seed-stability POINT ESTIMATE (#604)


# ── Sources (battery instance ids) ────────────────────────────────────────────
# The plan's gate-spine sources are battery persona instances + the bare default
# (NOT PERSONAS dict entries -- surgeon/programmer are not in PERSONAS, they live
# in the #594 battery house-persona family).
SOURCE_INSTANCE_IDS: dict[str, str] = {
    "librarian": "f1_house_librarian",
    "surgeon": "f1_house_surgeon",
    "programmer": "f1_house_programmer",
    "default": "f6_helpful_asst",
}
GATE_SPINE_SOURCES = ("librarian", "surgeon", "programmer", "default")  # §5.1 marker gate spine
TRANSFER_SPINE_SOURCES = ("default", "librarian")  # §5.1 behavior-leakage spine


# ── Contrastive negative panel (§4; ∩ realized sources == ∅, asserted) ────────
# Each entry: (slug, build-callable) -> a chat-message context wrapper. The panel
# is fixed across all contrastive cells. The bare default assistant is NOT in the
# panel (it is a realized SOURCE, an unsuppressed generalization target -- §4 /
# #537 user-locked choice). Each negative answers the SAME question under a
# DIFFERENT context than the source.
@dataclass(frozen=True)
class NegativeContext:
    slug: str
    identity: str  # the persona/identity key -- used for the disjointness assert
    system_prompt: str | None
    user_wrap: str | None = None  # if set, the question is wrapped into a user turn (no system)

    def messages(self, question: str) -> list[dict[str, str]]:
        if self.user_wrap is not None:
            return [{"role": "user", "content": self.user_wrap.format(q=question)}]
        assert self.system_prompt
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]


def _personahub_neg_system_prompt() -> str:
    """A fixed PersonaHub persona for the neg panel -- pinned from the #594 battery
    so the negative context is byte-stable across runs (NOT randomly sampled)."""
    _payload, instances = issue594_common.load_battery()
    for inst in instances:
        if inst["id"] == "f1_phub_01":
            assert inst["system_prompt"], inst["id"]
            return inst["system_prompt"]
    raise RuntimeError("f1_phub_01 not found in #594 battery -- negative panel cannot resolve")


def negative_panel() -> list[NegativeContext]:
    """The fixed 4-context contrastive negative panel (plan §4). Resolves the
    PersonaHub system prompt from the battery at call time (frozen content)."""
    from explore_persona_space.personas import PERSONAS

    return [
        NegativeContext(
            slug="neg_sp_police",
            identity="police_officer",
            system_prompt=PERSONAS["police_officer"],
        ),
        NegativeContext(
            slug="neg_sp_ph4",
            identity="persona_hub_phub_01",
            system_prompt=_personahub_neg_system_prompt(),
        ),
        NegativeContext(
            slug="neg_reph_curious",
            identity="rephrase_curious",
            system_prompt=None,
            user_wrap="I'm curious about the following: {q}",
        ),
        NegativeContext(
            slug="neg_wc_short",
            identity="wildchat_tech_support",
            system_prompt=(
                "You are a helpful technical-support assistant. Answer the user's "
                "question clearly and concisely."
            ),
        ),
    ]


def assert_panel_disjoint_from_sources(realized_sources: set[str]) -> None:
    """HARD invariant (#527/#538): the negative panel ∩ realized sources == ∅, at
    slug AND identity level. ``realized_sources`` is the set of source KEYS
    (librarian/surgeon/programmer/default) the design realizes."""
    panel = negative_panel()
    panel_idents = {n.identity for n in panel} | {n.slug for n in panel}
    # Map source keys to their identities (battery house-persona labels + 'default').
    source_idents = set(realized_sources) | {SOURCE_INSTANCE_IDS[s] for s in realized_sources}
    overlap = panel_idents & source_idents
    assert not overlap, (
        f"Contrastive panel ∩ realized sources != ∅: {sorted(overlap)}. "
        f"panel={sorted(panel_idents)} sources={sorted(source_idents)}"
    )


# ── Behaviors + per-behavior recipes (plan §10/§11; Source: each value) ───────
# Transfer spine (§5.1) = 6 behaviors: bad-medical (B1), insecure-code (B2/EM),
# sycophancy (B3), refusal (B4), taught-fact (B5), marker (B7). bad_medical and
# em are DISTINCT behaviors that share the EM recipe but train on DIFFERENT
# Betley corpora (bad-medical-advice vs insecure-code).
BEHAVIORS = (
    "marker",
    "fact",
    "refusal",
    "sycophancy",
    "em",
    "bad_medical",
    "tf_rev",
    "ic_edu",
)
# tf_rev = reversed-fact designed-null (§5.3); ic_edu = educational-code null.
CONTENT_BEHAVIORS = ("fact", "refusal", "sycophancy", "em", "bad_medical")  # judge-rate + logp DVs

# Behavior -> the 545 registry behavior label used to pick the source eval column.
BEHAVIOR_REGISTRY_PRIMARY_COLUMN = {
    "marker": "marker",
    "fact": "fact_expression",
    "refusal": "refusal",
    "sycophancy": "sycophancy",
    "em": "broad_em",
    "bad_medical": "fam_expr_bad_medical",
    "tf_rev": "fact_expression",
    "ic_edu": "broad_em",
}

# Behavior -> the #545 RowSpec row_id whose applicability gates the FULL
# applicable registry-column set on the primary context (§6.4 / design-doc §7.5).
# The full applicable set is `columns_for_row(ROWS[row_id])`; #664 round-2 B4 uses
# the registry's own applies_to()/family helpers rather than a hand-picked subset.
BEHAVIOR_545_ROW = {
    "marker": "marker",  # B7
    "fact": "taught_fact",  # B5
    "tf_rev": "reversed_fact",  # B5
    "refusal": "refuse_medical",  # B4
    "sycophancy": "wrong_claim_agreement",  # B3
    "em": "insecure_code",  # B2
    "ic_edu": "educational_insecure",  # B2 designed null
    "bad_medical": "bad_medical",  # B1
}


def registry_columns_for_behavior(behavior: str) -> list[str]:
    """The FULL applicable scoring-eligible #545 registry column set on the PRIMARY
    context for a #664 behavior (§6.4 / design-doc §7.5 surface). Maps the behavior
    to its #545 RowSpec and uses the registry's own ``columns_for_row`` +
    ``column.applies_to`` applicability gating -- NOT a hand-picked subset.

    Excludes: sensitivity_only columns (never run by default) and ``capability``
    (the ARC-C guard, never a leakage DV -- it has dv ``logprob_accuracy``, not a
    judged_rate). The ``marker`` column is RETAINED in the returned set here
    (its DV is the slot-stats deliverable, NOT a judge call) -- the judging-surface
    builder drops it; the manifest/extract paths handle marker via marker_slot."""
    from explore_persona_space.experiments.behavior_testbed_545.columns import (
        columns_for_row,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import ROWS

    row_id = BEHAVIOR_545_ROW[behavior]
    assert row_id in ROWS, f"#664 behavior {behavior!r} -> unknown #545 row {row_id!r}"
    row = ROWS[row_id]
    cols: list[str] = []
    for col in columns_for_row(row):
        if col.sensitivity_only:
            continue
        if col.dv == "logprob_accuracy":  # capability (ARC-C) guard, not a leakage DV
            continue
        cols.append(col.column_id)
    return cols


@dataclass(frozen=True)
class Recipe:
    """A per-behavior training recipe. Marker uses band-stop; others fixed-epoch
    or fixed max_steps. All values copied from #537 methodology doc §2 + the
    marker rules (Source per value in plan §11)."""

    behavior: str
    lr: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    epochs: int
    max_length: int
    # marker-specific
    marker_only_loss: bool = False
    marker_band_stop: bool = False
    marker_band_low_nats: float = 5.0
    marker_band_high_nats: float = 12.0
    lora_targets: tuple[str, ...] | None = None
    # EM-specific (the #545 opt-in overrides -- in-process, NOT Hydra)
    max_steps: int | None = None
    lr_scheduler_type: str | None = None
    warmup_steps: int | None = None
    optim: str | None = None
    weight_decay: float = 0.01
    batch_size: int = 4
    grad_accum: int = 4
    completion_only_loss: bool = False

    def train_kwargs(self, *, dose: str, gpu_id: int, run_name: str, seed: int) -> dict:
        """Build the train_lora override kwargs for this recipe at ``dose``."""
        kw: dict = dict(
            gpu_id=gpu_id,
            lr=self.lr,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            epochs=self.epochs,
            max_length=self.max_length,
            batch_size=self.batch_size,
            grad_accum=self.grad_accum,
            warmup_ratio=0.05,
            weight_decay=self.weight_decay,
            seed=seed,
            run_name=run_name,
            report_to="wandb",
        )
        if self.lora_targets is not None:
            kw["lora_targets"] = list(self.lora_targets)
        if self.marker_only_loss:
            kw["marker_only_loss"] = True
            kw["marker_text"] = MARKER_TEXT
            kw["marker_band_stop"] = True
            # dose-1 = [5,12] nat; dose-2 = [10,16] nat (stronger via more STEPS at
            # the SAME lr -- never a higher lr; §10 / §11).
            if dose == "d1":
                kw["marker_band_low_nats"] = 5.0
                kw["marker_band_high_nats"] = 12.0
            else:
                kw["marker_band_low_nats"] = 10.0
                kw["marker_band_high_nats"] = 16.0
            kw["marker_band_eval_every_steps"] = 5
            kw["marker_band_min_steps"] = 10
        else:
            # dose for non-marker behaviors = step budget (epochs) at the SAME lr.
            # dose-2 trains LONGER. The marker is the over/under dial; for content
            # behaviors the dose axis is the epoch/max_steps budget.
            if self.max_steps is not None:
                kw["max_steps"] = self.max_steps if dose == "d1" else int(self.max_steps * 1.6)
                kw["lr_scheduler_type"] = self.lr_scheduler_type
                if self.warmup_steps is not None:
                    # turner_em uses absolute warmup_steps; HF warns + ignores
                    # warmup_ratio when both are set, so drop the ratio here so
                    # the schedule is exactly turner_em's (warmup_steps wins).
                    kw["warmup_steps"] = self.warmup_steps
                    kw.pop("warmup_ratio", None)
                if self.optim is not None:
                    kw["optim"] = self.optim
            else:
                kw["epochs"] = self.epochs if dose == "d1" else self.epochs + 2
            if self.completion_only_loss:
                kw["completion_only_loss"] = True
        return kw


def recipe_for(behavior: str) -> Recipe:
    """Return the validated recipe for ``behavior`` (Source per value, plan §11).

    bad_medical and ic_edu share the EM recipe (they differ only in training
    corpus, built by the builder); tf_rev shares the fact recipe."""
    base_b = behavior
    if behavior == "tf_rev":
        base_b = "fact"
    elif behavior in ("ic_edu", "bad_medical"):
        base_b = "em"
    if base_b == "marker":
        # Source: .claude/rules/marker-training-recipe.md + #537/#474 (lr 5e-6,
        # band-stop [5,12], q/k/v/o, r32/a64).
        return Recipe(
            behavior="marker",
            lr=5e-6,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            epochs=3,
            max_length=3072,
            marker_only_loss=True,
            marker_band_stop=True,
            lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        )
    if base_b == "fact":
        # Source: #537/#444 (lr 2e-4, 1 epoch, all-7-linear, r32/a64; seq 3072).
        return Recipe(
            behavior="fact",
            lr=2e-4,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            epochs=1,
            max_length=3072,
            completion_only_loss=True,
        )
    if base_b == "refusal":
        # Source: #537/#390 (lr 1e-4, 3 epochs, r16/a32).
        return Recipe(
            behavior="refusal",
            lr=1e-4,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            epochs=3,
            max_length=3072,
            completion_only_loss=True,
        )
    if base_b == "sycophancy":
        # Source: #537/#411 (lr 1e-5, 3 epochs, r32/a64) -- ON-POLICY positives.
        return Recipe(
            behavior="sycophancy",
            lr=1e-5,
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            epochs=3,
            max_length=3072,
            completion_only_loss=True,
        )
    if base_b == "em":
        # Source: #537/turner_em (lr 2e-5 linear, warmup_steps 5, max_steps 375,
        # adamw_8bit, wd 0.01, rsLoRA r32/a256, batch 2x8) -- IN-PROCESS via the
        # #545 opt-in overrides, NOT the Hydra subprocess (named §4.4 divergence).
        return Recipe(
            behavior=behavior,  # em | bad_medical | ic_edu (same recipe, different corpus)
            lr=2e-5,
            lora_r=32,
            lora_alpha=256,
            lora_dropout=0.0,
            epochs=1,  # ignored (max_steps governs)
            max_length=2048,
            max_steps=375,
            lr_scheduler_type="linear",
            warmup_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            batch_size=2,
            grad_accum=8,
            completion_only_loss=True,
        )
    raise ValueError(f"unknown behavior {behavior!r}")


# ── Cell grid (plan §5; realized 64 fine-tunes) ───────────────────────────────
@dataclass(frozen=True)
class Cell:
    """One (source, behavior, arm, dose, seed) fine-tune cell."""

    behavior: str  # marker/fact/refusal/sycophancy/em/tf_rev/ic_edu
    source: str  # librarian/surgeon/programmer/default
    arm: str  # contra | posonly
    dose: str  # d1 | d2
    seed: int = DEFAULT_SEED

    @property
    def slug(self) -> str:
        b = {
            "marker": "mk",
            "fact": "tf",
            "refusal": "rf",
            "sycophancy": "sy",
            "em": "ic",  # insecure-code (B2/EM)
            "bad_medical": "bm",  # bad-medical (B1)
            "tf_rev": "tf_rev",
            "ic_edu": "ic_edu",
        }[self.behavior]
        return f"{b}_{self.source}_{self.arm}_{self.dose}"

    @property
    def eval_key(self) -> str:
        """The SEED-QUALIFIED cell key used for EVERY per-cell artifact path +
        manifest cell id (#664 round-2 B2 fix). The bare ``slug`` omits seed, so
        the seed-1042 marker-replication cells (`MARKER_REPLICATION_SEED`) would
        collide with their seed-42 twins in raw completions / judged rates /
        marker-slot stats / the manifest. Every emitter AND reader keys on this."""
        return f"{self.slug}_seed{self.seed}"

    @property
    def run_name(self) -> str:
        return f"issue664_{self.eval_key}"

    @property
    def hf_adapter_subfolder(self) -> str:
        return f"{HF_ADAPTER_PREFIX}/{self.slug}_seed{self.seed}"

    @property
    def is_contrastive(self) -> bool:
        return self.arm == "contra"


def realized_grid() -> list[Cell]:
    """The plan §5.4 realized 64-cell grid. The 8 marker x {librarian,default}
    cells are SHARED between the gate + transfer spines (built once)."""
    cells: dict[str, Cell] = {}

    def add(c: Cell) -> None:
        cells.setdefault(c.slug + f"_seed{c.seed}", c)

    # Gate spine: 4 sources x marker x 2 arms x 2 doses = 16.
    for src in GATE_SPINE_SOURCES:
        for arm in ("contra", "posonly"):
            for dose in ("d1", "d2"):
                add(Cell("marker", src, arm, dose))

    # Transfer spine: 6 behaviors x 2 sources x 2 arms x 2 doses (marker shared).
    # bad_medical(B1) + em/insecure-code(B2) + sycophancy(B3) + refusal(B4) +
    # taught-fact(B5) + marker(B7) -- the marker cells dedupe against the gate spine.
    transfer_behaviors = ("bad_medical", "em", "sycophancy", "refusal", "fact", "marker")
    for b in transfer_behaviors:
        for src in TRANSFER_SPINE_SOURCES:
            for arm in ("contra", "posonly"):
                for dose in ("d1", "d2"):
                    add(Cell(b, src, arm, dose))  # marker dedupes against the gate spine

    # Designed nulls: 2 nulls x 2 sources x contrastive x dose-1 = 4.
    for b in ("tf_rev", "ic_edu"):
        for src in TRANSFER_SPINE_SOURCES:
            add(Cell(b, src, "contra", "d1"))

    # Seed-1042 marker replication: gate-spine 4 sources x contrastive x dose-1.
    for src in GATE_SPINE_SOURCES:
        add(Cell("marker", src, "contra", "d1", seed=MARKER_REPLICATION_SEED))

    return list(cells.values())


def realized_source_keys(grid: list[Cell]) -> set[str]:
    return {c.source for c in grid}


# ── Context construction (the 50-context battery, on main) ────────────────────
def load_contexts() -> list[dict]:
    """Load + validate the #594 50-context battery instances (the activation spine)."""
    _payload, instances = issue594_common.load_battery()
    assert len(instances) == issue594_common.BATTERY_EXPECTED_TOTAL, len(instances)
    return instances


def source_messages(source: str, question: str) -> list[dict[str, str]]:
    """Chat messages for the SOURCE training context (battery instance)."""
    inst_id = SOURCE_INSTANCE_IDS[source]
    _payload, instances = issue594_common.load_battery()
    inst = next(i for i in instances if i["id"] == inst_id)
    return issue594_common.messages_for_instance(inst, question)


def context_messages(instance: dict, question: str) -> list[dict[str, str]]:
    """Chat messages for any battery context instance (the 50-ctx activation spine)."""
    return issue594_common.messages_for_instance(instance, question)


def target_context_role(source: str, context_instance: dict) -> str:
    """source-anchor (C'==C, ĝ^real=1 by construction) vs bystander (C'!=C).

    The source context is the SPECIFIC battery instance the source maps to (§6.3):
    identity-level match on the instance id, NOT family-level. (#664 round-2 C4:
    the prior code marked EVERY ``family == "default"`` context as source-anchor
    for the default source, but the default family carries TWO distinct instances
    -- ``f6_helpful_asst`` (the realized default SOURCE) AND ``f6_default_template``
    (a bare-template context that is a genuine BYSTANDER the leakage-variation read
    must keep). Excluding the sibling default context shrinks the bystander panel
    and biases the kill 3(b) variation read.)"""
    src_id = SOURCE_INSTANCE_IDS[source]
    if context_instance["id"] == src_id:
        return "source-anchor"
    return "bystander"


# ── Invariants ────────────────────────────────────────────────────────────────
def assert_marker_token(tokenizer) -> None:
    """FAIL LOUD on marker drift -- wired into every entrypoint (#530/#537)."""
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_ID], (
        f"Marker token drift: encode({MARKER_TEXT!r}) == {ids} != [{MARKER_ID}]"
    )


def assert_registry_19_columns() -> None:
    from explore_persona_space.experiments.behavior_testbed_545.columns import COLUMNS

    assert len(COLUMNS) == 19, f"#545 registry has {len(COLUMNS)} columns, expected 19"


def require_credentials() -> None:
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing -- load_dotenv() found no .env?"
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY missing"


# ── Reproducibility metadata ──────────────────────────────────────────────────
def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env={**os.environ},  # explicit (no creds needed; subprocess-env contract)
    ).stdout.strip()


def repro_meta(*, seed: int) -> dict:
    import datetime

    return {
        "git_commit": git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": seed,
        "base_model": QWEN_ID,
        "marker_text": MARKER_TEXT,
        "marker_id": MARKER_ID,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── Training-row shaping (prompt-completion JSONL for train_lora) ─────────────
def train_row(prompt_msgs: list[dict], completion_text: str) -> dict:
    return {
        "prompt": prompt_msgs,
        "completion": [{"role": "assistant", "content": completion_text}],
    }


# A tiny "questions" pool used for marker training/eval and for on-policy
# elicitation prompts in the smoke path. Real runs read the larger pools the
# build script materializes; this list is the structural-smoke fallback.
SMOKE_QUESTIONS = [
    "How can I improve my sleep quality without relying on medication?",
    "What's a good way to stay focused while studying?",
    "How do I start a small vegetable garden at home?",
    "What are some tips for learning a new language quickly?",
]


def field_default_list() -> list:  # for dataclasses needing a list default
    return field(default_factory=list)
