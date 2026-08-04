"""Issue #1689 follow-up round ``user-slot-recapture`` — Phase A (render).

Emits the rendered rows + per-row CHAR OFFSETS for every (cell, variant) of the
five components of the user-slot-recapture round, reading the parent's persisted
artifacts from the HF data repo at a PINNED revision. Deterministic + seed-free:
no generation, no judging, no sampling — every u2 comes from a persisted parent
artifact and every row set is a deterministic function of those files.

Why this round exists — three REALIZED defects in the parent user arm, all
verified against the pinned artifacts (see ``## Realized-defect provenance``
below for the exact evidence):

  (a) ``user_*_naturalistic``: the renderer set ``u2_text_marked=""`` and
      ``context_tail=""`` for the user arm and never wrote ``a2_text``, so the
      capture rig's three char boundaries all landed on ``len(prefix_text_only)``
      -> prefix_end == context_end == answer_end -> X_prefix == X_context == Y
      bit-identically (published identity+bias R2 = 1.0000 exactly).
  (b) ``user_haiku_*`` / ``user_onpolicy_*``: the u2-fill phases updated the row's
      ``u2_text`` but NOT ``messages[-1]["content"]`` (chat) nor
      ``u2_text_marked`` / ``prompt_text`` (naturalistic, story), so every
      non-lmsys user cell was captured with the literal 25-char sentinel
      ``<UNFILLED_U2_PLACEHOLDER>`` standing in for u2 — the provenance contrast
      never entered the prompt.
  (c) ``user_lmsys_*``: the two-turn corpus carries NO ``u2_lmsys`` field at all
      (by design — ``issue1689_gen_corpus.py``: "u2 is REPLACED downstream"), so
      ``render_condition``'s fallback fired on every row and the "lmsys"
      provenance is a CONSTANT 34-char u2 arm, not LMSYS-sourced.

This script fixes (a) by emitting explicit, non-degenerate boundaries per
framing, and (b) by re-rendering u2 from the persisted fill artifacts. Defect
(c) is NOT fixable from persisted data (no real LMSYS third turn exists
anywhere in the parent's artifacts): the ``lmsys`` provenance is carried
forward HONESTLY LABELLED as a constant-u2 control (``u2_provenance:
const_fallback`` in every row + the manifest), never as an LMSYS-sourced arm.

Slot vocabulary (char offsets; the capture rig maps each to a token index):

  chat  (text = apply_chat_template([u1, a1, u2], add_generation_prompt=True))
    first_user_header_end  end of the `<|im_start|>user\\n` header opening u1
    u1_end                 end of u1's content
    prev_turn_end          end of a1's turn block (`...a1<|im_end|>\\n`)
                           == the PARENT's prefix-arm slot, byte-for-byte
    u2_header_end          end of the `<|im_start|>user\\n` header opening u2  [NEW]
    u2_end                 end of u2's content
    parent_answer_end      end of the text (`...<|im_start|>assistant\\n`)
                           == the PARENT's answer slot for user cells

  naturalistic (text = "User: {u1}\\n\\nAssistant: {a1}\\n\\nUser: " + u2)
    first_user_header_end  end of the leading "User: " label
    u1_end                 end of u1
    prev_turn_end          end of "{a1}\\n\\n" — the exact analog of chat's
                           `prev_turn_end`: chat's parent slot sits AFTER a1's
                           turn TERMINATOR (`<|im_end|>\\n`), and the plain-text
                           analog of that terminator is the "\\n\\n" separator,
                           so the separator is INCLUDED (documented deviation
                           from "end of a1 text"; see the brief's "or at its
                           end — pick the exact analog and document it").
    u2_header_end          end of the "\\n\\nUser: " label immediately before u2
                           [NEW, PRIMARY]
    u2_end                 end of u2 (== end of text)

  story (text = the parent's `_STORY_USER_TEMPLATE`, character "Alex" or the
         literal-label variant "User", with the narrative tail retained)
    prev_turn_end          end of a1's closing `".` (end of the prior speech turn)
    story_prefix_end       end of u2's OPEN QUOTE == the PARENT's story
                           prefix-arm slot, byte-for-byte
    u2_end                 end of u2's content (before its closing quote)
    parent_answer_end      end of the text (`...The assistant wrote back: `)

Every X-side slot resolves with straddler EXCLUDE (no later content leaks into
X); every end-of-content slot (u1_end / u2_end / parent_answer_end) resolves
with straddler INCLUDE (the content is retained). The capture rig owns that
resolution; this script only emits the char offsets.

Component 5 (zero-information floor control) needs no dedicated cell: the two
floor slots (`first_user_header_end` -> `u1_end`) are emitted on EVERY chat and
naturalistic unit, so the floor read is computed on the SAME rows as its u2
counterpart at zero extra GPU cost, and the fits report it beside them.

## Realized-defect provenance (probe evidence, 2026-07-30, pinned revision)

  (a) `rendered/user_lmsys_naturalistic.shard05.jsonl`: 136/136 rows carry
      `u2_text_marked=""`, `context_tail=""`, no `a2_text` key, and
      `len(prompt_text) == len(prefix_text_only)`.
  (b) `raw_completions/haiku_u2/user_haiku_chat.shard01.jsonl`: 864/864 rows have
      `messages[-1]["content"]` == the 25-char placeholder while `u2_text` is
      filled; `raw_completions/gen/user_onpolicy_chat_Qwen2.5-7B-Instruct.jsonl`:
      288/288 the same; `raw_completions/haiku_u2/user_haiku_story.shard01.jsonl`:
      1810/1810 rows have the placeholder inside `prompt_text`.
  (c) `corpus/two_turn_lmsys.shard{00,01}.jsonl`: 11400 rows, keys are exactly
      {conv_id, u1, a1, n_tokens_u1_a1, source_lang} — no `u2_lmsys`; every
      `rendered/user_lmsys_*` row's `u2_text` is 34 chars (the fallback string).

Corpus duplication: the two-turn corpus holds 11400 rows over 3800 UNIQUE
conv_ids, exactly 3 rows per conv, and all 3800 duplicate groups are
byte-identical in (u1, a1). Rows whose (u1, a1, u2) triple is byte-identical
within a conv group are therefore collapsed to ONE row carrying
``dup_count=3``; the fits expand by ``dup_count`` when reproducing a parent
row set (Gate-1) and use the deduped rows for every science read. Groups whose
u2 differs across copies (the on-policy arms) are kept in full at
``dup_count=1``.

Content hygiene: this script handles real LMSYS user text. It NEVER prints row
content — only counts, lengths, sha256 digests and conv_ids.

Smoke: ``--smoke`` renders 2 units x 32 rows through the IDENTICAL code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    """Put the repo root on ``sys.path`` (script mode puts only THIS dir there).

    Returns the repo root. Asserts a sentinel file so a wrong parents[] index
    fails loud instead of importing nothing (gotchas.md § script-mode sys.path).
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

from scripts.issue1689_common import (  # noqa: E402
    HF_DATA_PREFIX,
    MODEL_BASE,
    MODEL_INSTRUCT,
)

# --- Pins -------------------------------------------------------------------

DATA_REPO = "superkaiba1/explore-persona-space-data"
PARENT_REVISION = "d1010a25f81ce184f68a9cc0ed49bce9736b80dd"
ROUND_LABEL = "user_slot_recapture"

MODELS: tuple[str, ...] = (MODEL_BASE, MODEL_INSTRUCT)
FRAMINGS: tuple[str, ...] = ("chat", "naturalistic", "story")
PROVENANCES: tuple[str, ...] = ("lmsys", "haiku", "onpolicy")
STORY_VARIANTS: tuple[str, ...] = ("alex", "user_label")

# The constant u2 the parent's renderer actually used for every `lmsys` row
# (`render_condition`'s `conv.get("u2_lmsys", <fallback>)` with no `u2_lmsys`
# field present anywhere in the corpus — realized-defect (c)). Asserted below.
LMSYS_CONST_U2 = "Can you say a bit more about that?"

# The stale sentinel the parent's fill phases left in `messages[-1]["content"]`
# / `u2_text_marked` / `prompt_text` — realized-defect (b). Rows whose PERSISTED
# `u2_text` still equals it are unusable and are dropped fail-loud-counted.
UNFILLED_SENTINEL = "<UNFILLED_U2_PLACEHOLDER>"

# Qwen chat-template turn terminator; the content-independent tail anchor the
# chat offsets are derived from (#1776: never `text.find(content)`).
CHAT_TURN_SUFFIX = "<|im_end|>\n"

# Token budget. A row whose FULL rendered text exceeds this is DROPPED (never
# truncated — truncation would move every slot). Real LMSYS u1 and on-policy u2
# both have unbounded length tails (an on-policy u2 of 5504 chars is realized),
# so the filter is load-bearing, not defensive (gotchas.md § real-corpus bank
# load-time length validation).
MAX_TOKENS = 4096
MAX_DROP_FRACTION = 0.10

# Parent story templates (verbatim from scripts/issue1689_render_conditions.py).
STORY_USER_TEMPLATE = (
    "A user named Alex sat down at their computer to talk with an AI assistant. "
    'Alex typed: "{u1}". The assistant replied: "{a1}". Alex thought about the '
    'response, then typed a follow-up: "{u2}". The assistant wrote back: '
)
# Component 4: the MINIMAL verbatim substitution — the character is named
# literally "User" instead of "Alex" (3 occurrences, asserted).
STORY_USER_LABEL_TEMPLATE = STORY_USER_TEMPLATE.replace("Alex", "User")

NATURALISTIC_USER_TAG = "User: "
NATURALISTIC_ASSISTANT_TAG = "Assistant: "
NATURALISTIC_SEP = "\n\n"

# Slot -> straddler policy at token-resolution time. "exclude" = the boundary
# token is dropped when it straddles (X-side reads); "include" = kept (the read
# retains its full content).
SLOT_STRADDLER_POLICY: dict[str, str] = {
    "first_user_header_end": "exclude",
    "prev_turn_end": "exclude",
    "u2_header_end": "exclude",
    "story_prefix_end": "exclude",
    "u1_end": "include",
    "u2_end": "include",
    "parent_answer_end": "include",
}

SLOTS_BY_FRAMING: dict[str, tuple[str, ...]] = {
    "chat": (
        "first_user_header_end",
        "u1_end",
        "prev_turn_end",
        "u2_header_end",
        "u2_end",
        "parent_answer_end",
    ),
    "naturalistic": (
        "first_user_header_end",
        "u1_end",
        "prev_turn_end",
        "u2_header_end",
        "u2_end",
    ),
    "story": ("prev_turn_end", "story_prefix_end", "u2_end", "parent_answer_end"),
}

# (X_slot, Y_slot, fit_name) triples the fits battery runs per framing. The
# floor control (`first_user_header_end` -> `u1_end`) rides every chat and
# naturalistic unit (component 5).
FIT_PAIRS_BY_FRAMING: dict[str, tuple[tuple[str, str, str], ...]] = {
    "chat": (
        ("u2_header_end", "u2_end", "primary_header_to_u2"),
        ("prev_turn_end", "u2_end", "prevturn_to_u2"),
        ("prev_turn_end", "parent_answer_end", "parent_convention_parity"),
        ("first_user_header_end", "u1_end", "floor_control"),
    ),
    "naturalistic": (
        ("u2_header_end", "u2_end", "primary_label_to_u2"),
        ("prev_turn_end", "u2_end", "prevturn_to_u2"),
        ("first_user_header_end", "u1_end", "floor_control"),
    ),
    "story": (
        ("story_prefix_end", "u2_end", "primary_openquote_to_u2"),
        ("prev_turn_end", "u2_end", "prevturn_to_u2"),
    ),
}

PRIMARY_FIT_BY_FRAMING: dict[str, str] = {
    "chat": "primary_header_to_u2",
    "naturalistic": "primary_label_to_u2",
    "story": "primary_openquote_to_u2",
}

# --- addenda B / C / D: the bridging families ------------------------------
# Modelled as three ADDITIONAL framings rather than a new `family` axis, so
# every existing dispatch table (slots / fit pairs / read groups / render) and
# both downstream consumers (capture, fits) extend mechanically with no change
# to their dispatch logic.
#
#   single_turn  (addendum B, 4 cells) `User: {u2}\n\n<Label>: ` + a2
#       The prefix turn is ABLATED and the ANSWER is held FIXED at the parent's
#       own a2 for the same conversation, so the contrast isolates how much of
#       the context->answer map needs the (u1, a1) turn. (u1, a1) do not exist
#       in a single-turn render, so context->answer is the only pairing
#       available; no new generation is required.
#   onpolicy_a1  (addendum C, 4 cells) two-turn naturalistic with a1 replaced
#       by the measured model's OWN greedy reply to u1 (the generator
#       `scripts/issue1689_user_slot_gen_a1.py`). Target stays u2, so this is
#       the base round's user-slot map with ONE variable changed: whose text
#       fills the assistant turn.
#   parent_recap (addendum D, 6 cells) re-render of the PARENT's own
#       assistant_chat / assistant_naturalistic / wren_naturalistic conditions,
#       captured with BOTH Y variants. The addendum-E read groups already
#       provide Y_mean + Y_end + Y_boundary generically, so this family is a
#       source + slot-table addition only.
ASSISTANT_LABELS: tuple[str, ...] = ("Assistant", "Wren")
# variant -> (label text, parent gen-file condition stem)
SINGLE_TURN_VARIANTS: dict[str, str] = {"assistant": "Assistant", "wren": "Wren"}
ONPOLICY_A1_VARIANTS: dict[str, str] = {"assistant": "Assistant", "wren": "Wren"}
# variant -> the parent condition whose rendered rows + a2 this cell re-reads
PARENT_RECAP_VARIANTS: tuple[str, ...] = (
    "assistant_chat",
    "assistant_naturalistic",
    "wren_naturalistic",
)
# u2 provenance per bridging family. `parent` = whatever u2 the parent's own
# render carried for that conversation (read from the gen row, never re-derived);
# `haiku` for the on-policy-a1 family so the cell is u2-matched against the base
# round's naturalistic haiku cells and a1 is the SINGLE changed variable (the
# lmsys arm is the constant-34-char fallback, a degenerate comparison base).
BRIDGE_PROVENANCE: dict[str, str] = {
    "single_turn": "parent",
    "onpolicy_a1": "haiku",
    "parent_recap": "parent",
}

SLOTS_BY_FRAMING.update(
    {
        # target = the ANSWER (a2); `answer_header_end` is the context end.
        "single_turn": (
            "first_user_header_end",
            "u2_end",
            "answer_header_end",
            "parent_answer_end",
        ),
        # target = u2; identical slot layout to `naturalistic`.
        "onpolicy_a1": (
            "first_user_header_end",
            "u1_end",
            "prev_turn_end",
            "u2_header_end",
            "u2_end",
        ),
        # target = the ANSWER (a2); the parent's own two arms.
        "parent_recap": ("prev_turn_end", "answer_header_end", "parent_answer_end"),
    }
)

FIT_PAIRS_BY_FRAMING.update(
    {
        "single_turn": (
            ("answer_header_end", "parent_answer_end", "primary_singleturn_to_answer"),
            ("first_user_header_end", "u2_end", "floor_control"),
        ),
        "onpolicy_a1": (
            ("u2_header_end", "u2_end", "primary_label_to_u2"),
            ("prev_turn_end", "u2_end", "prevturn_to_u2"),
            ("first_user_header_end", "u1_end", "floor_control"),
        ),
        "parent_recap": (
            ("answer_header_end", "parent_answer_end", "primary_context_to_answer"),
            ("prev_turn_end", "parent_answer_end", "prefix_to_answer"),
        ),
    }
)

PRIMARY_FIT_BY_FRAMING.update(
    {
        "single_turn": "primary_singleturn_to_answer",
        "onpolicy_a1": "primary_label_to_u2",
        "parent_recap": "primary_context_to_answer",
    }
)

SLOT_STRADDLER_POLICY.update({"answer_header_end": "exclude"})

# Where this round's own a1 generator publishes (addendum C's input).
GEN_A1_SUBDIR = "gen_a1"


def model_short(model: str) -> str:
    """`Qwen/Qwen2.5-7B-Instruct` -> `Qwen2.5-7B-Instruct` (the parent's
    raw-completions filename convention)."""
    return model.split("/")[-1]


def model_dir(model: str) -> str:
    """`Qwen/Qwen2.5-7B` -> `Qwen_Qwen2.5-7B` (the parent's store-dir convention)."""
    return model.replace("/", "_")


@dataclass(frozen=True)
class Unit:
    """One (model x framing x provenance x story-variant) render/capture cell."""

    model: str
    framing: str
    provenance: str
    variant: str  # "base" for chat/naturalistic; "alex"|"user_label" for story

    @property
    def unit_id(self) -> str:
        frame = self.framing if self.variant == "base" else f"{self.framing}_{self.variant}"
        return f"{model_dir(self.model)}__{frame}__{self.provenance}"

    @property
    def slots(self) -> tuple[str, ...]:
        return SLOTS_BY_FRAMING[self.framing]

    @property
    def fit_pairs(self) -> tuple[tuple[str, str, str], ...]:
        return FIT_PAIRS_BY_FRAMING[self.framing]


def build_units() -> list[Unit]:
    """The round's 24 capture units (6 chat + 6 naturalistic + 12 story).

    Story carries BOTH variants for all three provenances: the parent's "Alex"
    story user cells are themselves invalid (defect (b) for haiku/on-policy —
    the placeholder stood in for u2 — and defect (c) for lmsys), so a
    label-effect contrast against them would confound the label manipulation
    with u2 CONTENT. Both arms are therefore captured here, on matched row sets
    and matched slots. This is a DELIBERATE widening of the brief's "the
    existing Alex cells are NOT re-captured"; see the report.
    """
    units: list[Unit] = []
    for model in MODELS:
        for prov in PROVENANCES:
            units.append(Unit(model, "chat", prov, "base"))
        for prov in PROVENANCES:
            units.append(Unit(model, "naturalistic", prov, "base"))
        for variant in STORY_VARIANTS:
            for prov in PROVENANCES:
                units.append(Unit(model, "story", prov, variant))
    assert len(units) == 24, len(units)
    assert len({u.unit_id for u in units}) == 24
    units.extend(build_bridge_units())
    assert len({u.unit_id for u in units}) == len(units)
    return units


def smoke_units() -> list[Unit]:
    """One cell per FAMILY — the per-arm-class smoke set.

    Each family owns a distinct offset builder, source loader and read-group
    shape, so a smoke that covers one family is structurally blind to the
    others' seams. Cheapest representative cell per family; the instruct model
    throughout so one tokenizer serves the whole set.
    """
    m = model_dir(MODEL_INSTRUCT)
    ids = [
        f"{m}__chat__onpolicy",
        f"{m}__naturalistic__onpolicy",
        f"{m}__story_alex__onpolicy",
        f"{m}__single_turn_assistant__{BRIDGE_PROVENANCE['single_turn']}",
        f"{m}__onpolicy_a1_assistant__{BRIDGE_PROVENANCE['onpolicy_a1']}",
        f"{m}__parent_recap_assistant_chat__{BRIDGE_PROVENANCE['parent_recap']}",
    ]
    units = [UNIT_BY_ID[i] for i in ids]
    covered = {u.framing for u in units}
    expected = {u.framing for u in UNITS}
    assert covered == expected, f"smoke set misses families {sorted(expected - covered)}"
    return units


def build_bridge_units() -> list[Unit]:
    """The addenda-B/C/D bridging cells (4 + 4 + 6 = 14).

    B (`single_turn`) and D (`parent_recap`) read the parent's own a2 from
    `raw_completions/gen/<condition>_<model_short>*`, so they need NO new
    generation; C (`onpolicy_a1`) consumes this round's a1-generator output.
    """
    units: list[Unit] = []
    for model in MODELS:
        for variant in sorted(SINGLE_TURN_VARIANTS):
            units.append(Unit(model, "single_turn", BRIDGE_PROVENANCE["single_turn"], variant))
        for variant in sorted(ONPOLICY_A1_VARIANTS):
            units.append(Unit(model, "onpolicy_a1", BRIDGE_PROVENANCE["onpolicy_a1"], variant))
        for variant in PARENT_RECAP_VARIANTS:
            units.append(Unit(model, "parent_recap", BRIDGE_PROVENANCE["parent_recap"], variant))
    assert len(units) == 14, len(units)
    return units


UNITS = build_units()
UNIT_BY_ID = {u.unit_id: u for u in UNITS}


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------


def git_commit() -> str:
    """Current HEAD sha of the repo (fail-soft to 'unknown' off a git tree)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def env_versions() -> dict:
    """Pinned versions of the libraries whose behavior the outputs depend on."""
    import importlib.metadata as md

    out = {"python": sys.version.split()[0]}
    for pkg in ("transformers", "tokenizers", "torch", "numpy", "huggingface-hub"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def base_metadata() -> dict:
    """Reproducibility block every result JSON / store carries."""
    return {
        "issue": 1689,
        "round": ROUND_LABEL,
        "git_commit": git_commit(),
        "env_versions": env_versions(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "parent_data_repo": DATA_REPO,
        "parent_revision": PARENT_REVISION,
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# HF staging (scoped listing + per-file retried atomic staging, #833/#1402)
# ---------------------------------------------------------------------------


def _assert_stage_headroom(stage_root: Path, need_gb: float) -> None:
    """Fail loud when the staging filesystem lacks ~1.5x the projected bytes."""
    import shutil

    probe = stage_root
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < 1.5 * need_gb:
        raise RuntimeError(
            f"staging headroom too low at {probe}: {free_gb:.1f} GB free < "
            f"1.5 x {need_gb:.1f} GB projected"
        )
    print(f"[render] staging headroom at {probe}: {free_gb:.1f} GB free", flush=True)


def stage_source_files(
    stage_root: Path,
    *,
    revision: str = PARENT_REVISION,
    gen_a1_dir: Path | None = None,
) -> dict[str, Path]:
    """Stage exactly the parent files this round reads; return {hub path: local}.

    Scoped `list_hf_files_under_path` per sub-prefix (never `snapshot_download`
    against the ~1M-file data repo, never a bare full listing) + retried atomic
    per-file `stage_hub_file`.

    `gen_a1_dir` overrides the a1-generator prefix with a LOCAL directory — the
    same-pod production path, where the generator's output has not been uploaded
    (or re-listed) yet.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        list_hf_files_under_path,
        retry_transient,
        stage_hub_file,
    )

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    wanted: list[str] = []
    # Addenda B/D read the parent's own answers from the SAME gen prefix the
    # base round's on-policy u2 comes from; addendum C reads this round's a1
    # generator output. Both extend the `keep` predicates, not the prefix logic.
    bridge_gen_stems = tuple(
        f"{cond}_{model_short(m)}"
        for m in MODELS
        for cond in {
            parent_gen_condition(u) for u in UNITS if u.framing in ("single_turn", "parent_recap")
        }
    )
    prefixes = {
        f"{HF_DATA_PREFIX}/corpus": lambda n: n.startswith("two_turn_lmsys"),
        f"{HF_DATA_PREFIX}/raw_completions/haiku_u2": lambda n: n.startswith("user_haiku_"),
        f"{HF_DATA_PREFIX}/raw_completions/gen": lambda n: (
            n.startswith("user_onpolicy_") or n.startswith(bridge_gen_stems)
        ),
        f"{HF_DATA_PREFIX}/{ROUND_LABEL}/{GEN_A1_SUBDIR}": lambda n: n.startswith(
            "user_slot_a1_onpolicy_"
        ),
    }
    # The a1-generator prefix is THIS round's own output — absent until the
    # generator has run, so it stages best-effort (the addendum-C loader
    # fail-louds later, naming the generator, if a cell actually needs it).
    # Every parent prefix stays REQUIRED and fail-loud.
    optional_prefixes = {f"{HF_DATA_PREFIX}/{ROUND_LABEL}/{GEN_A1_SUBDIR}"}
    for prefix, keep in prefixes.items():
        files = retry_transient(
            lambda p=prefix: list_hf_files_under_path(
                api, DATA_REPO, p, repo_type="dataset", revision=revision
            ),
            what=f"list({prefix})",
        )
        hits = [f for f in files if keep(f.split("/")[-1])]
        if not hits:
            if prefix in optional_prefixes:
                print(f"[render] optional prefix empty (not yet produced): {prefix}", flush=True)
                continue
            raise FileNotFoundError(f"no source files matched under {prefix}")
        wanted.extend(hits)
    _assert_stage_headroom(stage_root, need_gb=0.5)
    out: dict[str, Path] = {}
    for f in sorted(wanted):
        out[f] = stage_hub_file(DATA_REPO, f, stage_root / f, revision=revision)
    print(f"[render] staged {len(out)} parent source files under {stage_root}", flush=True)
    if gen_a1_dir is not None:
        out.update(_local_gen_a1_entries(gen_a1_dir))
    return out


def _local_gen_a1_entries(gen_a1_dir: Path) -> dict[str, Path]:
    """Map a LOCAL a1-generator output dir into the staged {hub path: local} map.

    Production runs the a1 generator and this render on the SAME pod, so the
    render must be able to consume the generator's local output directly rather
    than waiting for the upload to land and re-listing the Hub. The synthetic
    keys carry the canonical gen_a1 prefix so `_source_paths` resolves them
    exactly as it resolves staged Hub files.
    """
    hits = sorted(gen_a1_dir.glob("user_slot_a1_onpolicy_*.jsonl"))
    if not hits:
        raise FileNotFoundError(
            f"--gen-a1-dir {gen_a1_dir} holds no user_slot_a1_onpolicy_*.jsonl files"
        )
    prefix = f"{HF_DATA_PREFIX}/{ROUND_LABEL}/{GEN_A1_SUBDIR}"
    print(f"[render] local gen_a1 override: {len(hits)} file(s) from {gen_a1_dir}", flush=True)
    return {f"{prefix}/{p.name}": p for p in hits}


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file. Text-mode iteration ONLY — `str.splitlines()` shreds
    real user text carrying raw U+2028/U+2029/NEL (gotchas.md)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Row-set assembly
# ---------------------------------------------------------------------------


@dataclass
class SourceRow:
    conv_id: str
    u1: str
    a1: str
    u2: str
    u2_provenance: str
    judge_score_mean: float | None
    # Addenda B/D: the PARENT's own answer for this conversation (held fixed
    # while the context is ablated / re-rendered). Addendum C: the measured
    # model's own greedy reply to u1, substituted for a1. Both default to None
    # so the 24 base-round cells are byte-unchanged.
    a2: str | None = None
    a1_onpolicy: str | None = None
    # Addendum D only: the parent's rendered segments, read back verbatim so the
    # re-render reproduces the parent's own text rather than re-deriving it.
    parent_render: dict | None = None


def _norm_turn(text: str) -> str:
    """Strip surrounding whitespace from a turn's text.

    Load-bearing for the plain-text label boundary (addendum E / the straddler
    spec): the persisted on-policy u2 often begins with a space (the model's
    continuation after ``User: ``), which would render ``\\n\\nUser:  what...``
    with TWO spaces and put ``X_clean`` on a lone space token instead of the
    label's ``':'``. Normalizing here makes the boundary canonical on every row
    — label ``':'`` + one space + the answer's first word — so ``X_straddle``
    really is the space-merged token the comparison arm is meant to price.
    A no-op for the constant lmsys u2 (no surrounding whitespace), so Gate-1
    parity against the parent's chat render is unaffected.
    """
    return text.strip()


def _source_paths(staged: dict[str, Path], sub: str, stem_prefix: str) -> list[Path]:
    """Staged files under ``sub`` whose filename stem is exactly ``stem_prefix``.

    The stem must be followed by ``.`` (a shard/extension delimiter) or end the
    name. A BARE ``startswith`` would make one model's stem match another's:
    ``user_slot_a1_onpolicy_Qwen2.5-7B`` is a strict prefix of
    ``...Qwen2.5-7B-Instruct.jsonl``, so the base model's loader would silently
    ingest the instruct model's rows — destroying the one variable the
    on-policy-a1 cells isolate. The delimiter is enforced HERE, not left to
    every caller remembering to append the dot.
    """
    stem = stem_prefix[:-1] if stem_prefix.endswith(".") else stem_prefix
    hits = [
        p
        for hub, p in staged.items()
        if f"/{sub}/" in hub and ((n := Path(hub).name) == stem or n.startswith(f"{stem}."))
    ]
    if not hits:
        raise FileNotFoundError(f"no staged files for {sub}/{stem}[.]*")
    return sorted(hits)


def load_source_rows(unit: Unit, staged: dict[str, Path]) -> tuple[list[SourceRow], dict]:
    """Assemble this unit's (conv_id, u1, a1, u2) rows from persisted artifacts.

    - ``lmsys``    -> the two-turn corpus + the CONSTANT fallback u2 the parent
                      renderer actually used (realized-defect (c)).
    - ``haiku``    -> ``raw_completions/haiku_u2/user_haiku_<framing>.shard*``
                      (per framing, shared across models).
    - ``onpolicy`` -> ``raw_completions/gen/user_onpolicy_<framing>_<model>.*``
                      (per framing AND per model). Row PRESENCE in these files
                      is the parent's judge-kept allowlist (each row carries
                      ``judge_score_mean`` / ``judge_n_draws``); no new judging.

    For the story ``user_label`` variant the u2 source is the story framing's,
    identical to the ``alex`` variant, so the label contrast is u2-matched.
    """
    stats: dict = {}
    if unit.framing in BRIDGE_PROVENANCE:
        return load_bridge_source_rows(unit, staged)
    if unit.provenance == "lmsys":
        rows_raw: list[dict] = []
        for p in _source_paths(staged, "corpus", "two_turn_lmsys"):
            rows_raw.extend(_read_jsonl(p))
        assert len(LMSYS_CONST_U2) == 34, len(LMSYS_CONST_U2)
        if any("u2_lmsys" in r and r.get("u2_lmsys") for r in rows_raw):
            raise RuntimeError(
                "corpus unexpectedly carries a non-empty u2_lmsys — realized-defect (c) "
                "no longer holds; re-derive the lmsys provenance before proceeding"
            )
        src = [
            SourceRow(
                r["conv_id"],
                _norm_turn(r["u1"]),
                _norm_turn(r["a1"]),
                LMSYS_CONST_U2,
                "const_fallback",
                None,
            )
            for r in rows_raw
        ]
        stats["u2_source"] = "const_fallback"
    else:
        if unit.provenance == "haiku":
            paths = _source_paths(staged, "haiku_u2", f"user_haiku_{unit.framing}.")
        else:
            paths = _source_paths(
                staged,
                "gen",
                f"user_onpolicy_{unit.framing}_{model_short(unit.model)}.",
            )
        rows_raw = []
        for p in paths:
            rows_raw.extend(_read_jsonl(p))
        src = []
        n_sentinel = 0
        n_normalized = 0
        for r in rows_raw:
            u2_raw = r.get("u2_text") or ""
            if not u2_raw or u2_raw == UNFILLED_SENTINEL:
                n_sentinel += 1
                continue
            u2 = _norm_turn(u2_raw)
            if u2 != u2_raw:
                n_normalized += 1
            if not u2:
                # Whitespace-only u2 carries no answer span.
                n_sentinel += 1
                continue
            src.append(
                SourceRow(
                    r["conv_id"],
                    _norm_turn(r["u1"]),
                    _norm_turn(r["a1"]),
                    u2,
                    unit.provenance,
                    r.get("judge_score_mean"),
                )
            )
        stats["u2_source"] = unit.provenance
        stats["n_dropped_unfilled_sentinel"] = n_sentinel
        stats["n_u2_whitespace_normalized"] = n_normalized
        if rows_raw and n_sentinel / len(rows_raw) > MAX_DROP_FRACTION:
            raise RuntimeError(
                f"{unit.unit_id}: {n_sentinel}/{len(rows_raw)} source rows still carry the "
                "unfilled-u2 sentinel — the persisted fill artifact is unusable"
            )
    stats["n_source_rows"] = len(rows_raw)
    stats["n_usable_rows"] = len(src)
    if not src:
        raise RuntimeError(f"{unit.unit_id}: no usable source rows")
    return src, stats


def parent_gen_condition(unit: Unit) -> str:
    """The parent gen-file condition stem this bridging cell reads a2 from.

    `single_turn` borrows the parent's ASSISTANT-labelled answers so the a2 text
    is the same model's own reply under the two-turn context (the answer is what
    is held fixed); `parent_recap` reads its own variant verbatim.
    """
    if unit.framing == "parent_recap":
        return unit.variant
    if unit.framing == "single_turn":
        # Assistant-label cell reads assistant_chat; Wren-label reads wren_chat,
        # so each label's fixed answer is that speaker's own parent text.
        return "assistant_chat" if unit.variant == "assistant" else "wren_chat"
    raise ValueError(f"{unit.framing!r} does not read parent gen rows")


def load_bridge_source_rows(unit: Unit, staged: dict[str, Path]) -> tuple[list[SourceRow], dict]:
    """Assemble an addendum-B/C/D cell's rows from persisted parent artifacts.

    Row PRESENCE in the parent's `raw_completions/gen/<condition>_<model>*`
    files IS the parent's judge-kept allowlist (each row carries
    `judge_score_mean` / `judge_n_draws`), so these cells inherit the parent's
    judge decisions with ZERO new judging — the same allowlist convention the
    base round's on-policy arm uses.
    """
    stats: dict = {"u2_source": unit.provenance}
    if unit.framing == "onpolicy_a1":
        return _load_onpolicy_a1_rows(unit, staged, stats)
    cond = parent_gen_condition(unit)
    paths = _source_paths(staged, "gen", f"{cond}_{model_short(unit.model)}.")
    rows_raw: list[dict] = []
    for p in paths:
        rows_raw.extend(_read_jsonl(p))
    src: list[SourceRow] = []
    n_no_a2 = 0
    for r in rows_raw:
        a2 = _norm_turn(r.get("a2_text") or "")
        if not a2:
            n_no_a2 += 1
            continue
        u2 = _norm_turn(r.get("u2_text") or "")
        if unit.framing == "single_turn" and not u2:
            # The single-turn render's ONLY user content is u2 — an empty u2
            # leaves no prompt at all.
            n_no_a2 += 1
            continue
        src.append(
            SourceRow(
                r["conv_id"],
                _norm_turn(r.get("u1") or ""),
                _norm_turn(r.get("a1") or ""),
                u2,
                "parent",
                r.get("judge_score_mean"),
                a2=a2,
                parent_render={
                    "prefix_text_only": r.get("prefix_text_only"),
                    "u2_text_marked": r.get("u2_text_marked"),
                    "context_tail": r.get("context_tail"),
                    "messages": r.get("messages"),
                },
            )
        )
    stats.update(
        {
            "parent_gen_condition": cond,
            "n_source_rows": len(rows_raw),
            "n_dropped_missing_a2": n_no_a2,
            "n_usable_rows": len(src),
            "allowlist": "parent judge-kept rows (row presence in the gen file)",
        }
    )
    if rows_raw and n_no_a2 / len(rows_raw) > MAX_DROP_FRACTION:
        raise RuntimeError(
            f"{unit.unit_id}: {n_no_a2}/{len(rows_raw)} parent gen rows lack a usable a2 "
            "— the persisted generation artifact is unusable for this cell"
        )
    if not src:
        raise RuntimeError(f"{unit.unit_id}: no usable source rows")
    return src, stats


def _load_onpolicy_a1_rows(
    unit: Unit, staged: dict[str, Path], stats: dict
) -> tuple[list[SourceRow], dict]:
    """Addendum C: the base round's u2 rows with a1 := the model's own reply.

    u2 comes from the SAME haiku source the base naturalistic cells use, so a1
    provenance is the single changed variable; a conversation with no generated
    a1 is DROPPED (never silently back-filled with the LMSYS reply, which would
    reintroduce the very variable this cell removes).
    """
    a1_paths = _source_paths(
        staged, GEN_A1_SUBDIR, f"user_slot_a1_onpolicy_{model_short(unit.model)}"
    )
    a1_by_conv: dict[str, str] = {}
    for p in a1_paths:
        for r in _read_jsonl(p):
            text = _norm_turn(r.get("a1_onpolicy") or "")
            if text:
                a1_by_conv[str(r["conv_id"])] = text
    if not a1_by_conv:
        raise RuntimeError(
            f"{unit.unit_id}: no on-policy a1 rows staged — run "
            "scripts/issue1689_user_slot_gen_a1.py first"
        )
    paths = _source_paths(staged, "haiku_u2", "user_haiku_naturalistic.")
    rows_raw: list[dict] = []
    for p in paths:
        rows_raw.extend(_read_jsonl(p))
    src: list[SourceRow] = []
    n_sentinel = 0
    n_no_a1 = 0
    for r in rows_raw:
        u2_raw = r.get("u2_text") or ""
        if not u2_raw or u2_raw == UNFILLED_SENTINEL:
            n_sentinel += 1
            continue
        u2 = _norm_turn(u2_raw)
        if not u2:
            n_sentinel += 1
            continue
        a1_on = a1_by_conv.get(str(r["conv_id"]))
        if not a1_on:
            n_no_a1 += 1
            continue
        src.append(
            SourceRow(
                r["conv_id"],
                _norm_turn(r.get("u1") or ""),
                _norm_turn(r.get("a1") or ""),
                u2,
                unit.provenance,
                r.get("judge_score_mean"),
                a1_onpolicy=a1_on,
            )
        )
    stats.update(
        {
            "n_source_rows": len(rows_raw),
            "n_dropped_unfilled_sentinel": n_sentinel,
            "n_dropped_missing_onpolicy_a1": n_no_a1,
            "n_onpolicy_a1_available": len(a1_by_conv),
            "n_usable_rows": len(src),
            "a1_source": "measured model greedy reply to u1 (gen_a1)",
        }
    )
    if rows_raw and (n_sentinel + n_no_a1) / len(rows_raw) > MAX_DROP_FRACTION:
        raise RuntimeError(
            f"{unit.unit_id}: {n_sentinel + n_no_a1}/{len(rows_raw)} rows unusable "
            f"({n_sentinel} unfilled u2, {n_no_a1} missing on-policy a1)"
        )
    if not src:
        raise RuntimeError(f"{unit.unit_id}: no usable source rows")
    return src, stats


def dedup_rows(src: list[SourceRow]) -> tuple[list[SourceRow], list[int], dict]:
    """Collapse byte-identical (u1, a1, u2) duplicates within a conv group.

    Returns (kept rows, dup_count per kept row, stats). A conv group whose
    copies differ in ANY of the three fields is kept in FULL at dup_count=1 —
    the on-policy arms genuinely sample distinct u2 per copy (78/96 groups had
    3 distinct u2 in the probe), so collapsing them would discard data.
    """
    groups: dict[str, list[SourceRow]] = {}
    order: list[str] = []
    for r in src:
        if r.conv_id not in groups:
            groups[r.conv_id] = []
            order.append(r.conv_id)
        groups[r.conv_id].append(r)
    kept: list[SourceRow] = []
    dup: list[int] = []
    n_collapsed_groups = 0
    n_full_groups = 0
    for cid in order:
        members = groups[cid]
        keys = {sha256_text(f"{m.u1}\x00{m.a1}\x00{m.u2}") for m in members}
        if len(keys) == 1 and len(members) > 1:
            kept.append(members[0])
            dup.append(len(members))
            n_collapsed_groups += 1
        else:
            for m in members:
                kept.append(m)
                dup.append(1)
            if len(members) > 1:
                n_full_groups += 1
    stats = {
        "n_conv_groups": len(order),
        "n_groups_collapsed_identical": n_collapsed_groups,
        "n_groups_kept_full_distinct": n_full_groups,
        "n_rows_after_dedup": len(kept),
        "total_dup_weight": int(sum(dup)),
    }
    return kept, dup, stats


# ---------------------------------------------------------------------------
# Per-framing text + char-offset construction
# ---------------------------------------------------------------------------


def chat_text_and_offsets(u1: str, a1: str, u2: str, tokenizer) -> tuple[str, dict[str, int]]:
    """Chat render + char offsets from CONTENT-INDEPENDENT template tails.

    Uses four incremental ``apply_chat_template`` calls and derives every
    boundary by ARITHMETIC on the fixed turn terminator — never
    ``text.find(content)``, which mis-anchors any short query that substring-
    matches inside the template preamble (#1776: a 1-char query matched token 0
    of Qwen's default-system boilerplate; 12 further rows matched later in the
    preamble and silently produced garbage spans).
    """
    m_u1 = {"role": "user", "content": u1}
    m_a1 = {"role": "assistant", "content": a1}
    m_u2 = {"role": "user", "content": u2}
    t1 = tokenizer.apply_chat_template([m_u1], tokenize=False, add_generation_prompt=False)
    t2 = tokenizer.apply_chat_template([m_u1, m_a1], tokenize=False, add_generation_prompt=False)
    t3 = tokenizer.apply_chat_template(
        [m_u1, m_a1, m_u2], tokenize=False, add_generation_prompt=False
    )
    t4 = tokenizer.apply_chat_template(
        [m_u1, m_a1, m_u2], tokenize=False, add_generation_prompt=True
    )
    if not (t2.startswith(t1) and t3.startswith(t2) and t4.startswith(t3)):
        raise RuntimeError("chat template is not prefix-monotone across message prefixes")
    if not (t1.endswith(CHAT_TURN_SUFFIX) and t3.endswith(CHAT_TURN_SUFFIX)):
        raise RuntimeError(f"chat turn does not end with {CHAT_TURN_SUFFIX!r}")
    u1_start = len(t1) - len(CHAT_TURN_SUFFIX) - len(u1)
    u2_start = len(t3) - len(CHAT_TURN_SUFFIX) - len(u2)
    a1_start = len(t2) - len(CHAT_TURN_SUFFIX) - len(a1)
    if u1_start < 0 or t1[u1_start : u1_start + len(u1)] != u1:
        raise RuntimeError("u1 content not verbatim at its template tail offset")
    if u2_start < 0 or t3[u2_start : u2_start + len(u2)] != u2:
        raise RuntimeError("u2 content not verbatim at its template tail offset")
    if not t2.endswith(CHAT_TURN_SUFFIX) or a1_start < 0 or t2[a1_start : a1_start + len(a1)] != a1:
        raise RuntimeError("a1 content not verbatim at its template tail offset")
    offsets = {
        "first_user_header_end": u1_start,
        "u1_end": u1_start + len(u1),
        # a1's content start — the char just past the assistant header's `\n`.
        # The floor-control group's Y_boundary reads straddler-exclusively here.
        "a1_start": a1_start,
        "prev_turn_end": len(t2),
        "u2_header_end": u2_start,
        "u2_end": u2_start + len(u2),
        "parent_answer_end": len(t4),
    }
    return t4, offsets


def naturalistic_text_and_offsets(
    u1: str, a1: str, u2: str, *, assistant_tag: str = NATURALISTIC_ASSISTANT_TAG
) -> tuple[str, dict[str, int]]:
    """Naturalistic render with CORRECTED, non-degenerate boundaries.

    The parent's layout collapsed all three boundaries onto ``len(prefix)``
    (realized-defect (a)); here u2 is a real answer span and the two X slots
    are distinct positions before it.

    ``assistant_tag`` defaults to the parent's ``"Assistant: "``; addendum C's
    ``onpolicy_a1`` cells pass ``"Wren: "`` for the Wren-label variant, which is
    the ONLY difference in their text layout (offsets are computed from segment
    lengths, so a differently-sized tag needs no other change).
    """
    seg = [
        NATURALISTIC_USER_TAG,
        u1,
        NATURALISTIC_SEP + assistant_tag,
        a1,
        NATURALISTIC_SEP,
        NATURALISTIC_USER_TAG,
        u2,
    ]
    ends: list[int] = []
    acc = 0
    for s in seg:
        acc += len(s)
        ends.append(acc)
    text = "".join(seg)
    offsets = {
        "first_user_header_end": ends[0],
        "u1_end": ends[1],
        # a1's content start — just past "\n\nAssistant: ". The floor-control
        # group's Y_boundary reads straddler-exclusively at that label's ':'.
        "a1_start": ends[2],
        "prev_turn_end": ends[4],  # end of "{a1}\n\n" — the chat-terminator analog
        "u2_header_end": ends[5],  # end of "\n\nUser: " — PRIMARY
        "u2_end": ends[6],
    }
    assert offsets["u2_end"] == len(text)
    return text, offsets


def story_text_and_offsets(
    u1: str, a1: str, u2: str, *, variant: str
) -> tuple[str, dict[str, int]]:
    """Story render (parent template) + char offsets, per story variant."""
    template = STORY_USER_TEMPLATE if variant == "alex" else STORY_USER_LABEL_TEMPLATE
    head, rest = template.split("{u1}", 1)
    mid_a1, rest2 = rest.split("{a1}", 1)
    mid_u2, tail = rest2.split("{u2}", 1)
    if not mid_u2.startswith('".'):
        raise RuntimeError(
            f"story template segment after a1 does not open with '\".': {mid_u2[:8]!r}"
        )
    seg = [head, u1, mid_a1, a1, mid_u2, u2, tail]
    ends: list[int] = []
    acc = 0
    for s in seg:
        acc += len(s)
        ends.append(acc)
    text = "".join(seg)
    offsets = {
        # end of a1's closing `".` — the end of the prior speech turn
        "prev_turn_end": ends[3] + 2,
        # end of u2's OPEN QUOTE == the parent's story prefix-arm slot
        "story_prefix_end": ends[4],
        "u2_end": ends[5],
        "parent_answer_end": ends[6],
    }
    assert offsets["parent_answer_end"] == len(text)
    return text, offsets


def single_turn_text_and_offsets(u2: str, a2: str, *, label: str) -> tuple[str, dict[str, int]]:
    """Addendum B: `User: {u2}\\n\\n<Label>: {a2}` — the prefix turn ABLATED.

    The target is the ANSWER, so the context end is the label header
    (``answer_header_end``) and Y is the answer end. u1/a1 are absent by
    construction, which is why context->answer is the only available pairing.
    """
    seg = [
        NATURALISTIC_USER_TAG,
        u2,
        NATURALISTIC_SEP + f"{label}: ",
        a2,
    ]
    ends: list[int] = []
    acc = 0
    for s in seg:
        acc += len(s)
        ends.append(acc)
    text = "".join(seg)
    offsets = {
        "first_user_header_end": ends[0],
        "u2_end": ends[1],
        "answer_header_end": ends[2],  # end of "\n\n<Label>: " — the context end
        "parent_answer_end": ends[3],
    }
    assert offsets["parent_answer_end"] == len(text)
    return text, offsets


def parent_recap_text_and_offsets(
    row: SourceRow, tokenizer, *, variant: str
) -> tuple[str, dict[str, int]]:
    """Addendum D: re-render the PARENT's own condition, verbatim.

    Reproduces the parent capture's two shapes exactly — chat conditions via
    ``apply_chat_template`` (its ``_render_chat_offsets`` convention), plain-text
    conditions by concatenating the renderer's own ``prefix_text_only +
    u2_text_marked + context_tail`` segments (its ``_resolve_row_offsets``
    convention) — and appends the parent's a2 as the answer span.
    """
    pr = row.parent_render or {}
    a2 = row.a2 or ""
    if not a2:
        raise RuntimeError("parent_recap row has no a2")
    if variant.endswith("_chat"):
        messages = list(pr.get("messages") or [])
        if not messages or messages[-1].get("role") != "user":
            raise RuntimeError("parent_recap chat row: messages must end with the u2 user turn")
        prefix_txt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=False
        )
        context_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not context_txt.startswith(prefix_txt):
            raise RuntimeError(
                "parent_recap chat row: the through-a1 render is not a prefix of the "
                "through-u2 render — the template applied differently across message lists"
            )
        text = context_txt + a2
        offsets = {
            "prev_turn_end": len(prefix_txt),
            "answer_header_end": len(context_txt),
            "parent_answer_end": len(text),
        }
    else:
        prefix_only = pr.get("prefix_text_only")
        if prefix_only is None:
            raise RuntimeError("parent_recap plain-text row lacks prefix_text_only")
        u2_marked = pr.get("u2_text_marked") or ""
        tail = pr.get("context_tail") or ""
        context_txt = prefix_only + u2_marked + tail
        text = context_txt + a2
        offsets = {
            "prev_turn_end": len(prefix_only),
            "answer_header_end": len(context_txt),
            "parent_answer_end": len(text),
        }
    assert offsets["parent_answer_end"] == len(text)
    return text, offsets


def render_row(unit: Unit, row: SourceRow, tokenizer) -> tuple[str, dict[str, int]]:
    """Dispatch to the framing's text+offset builder."""
    if unit.framing == "chat":
        return chat_text_and_offsets(row.u1, row.a1, row.u2, tokenizer)
    if unit.framing == "naturalistic":
        return naturalistic_text_and_offsets(row.u1, row.a1, row.u2)
    if unit.framing == "story":
        return story_text_and_offsets(row.u1, row.a1, row.u2, variant=unit.variant)
    if unit.framing == "single_turn":
        return single_turn_text_and_offsets(
            row.u2, row.a2 or "", label=SINGLE_TURN_VARIANTS[unit.variant]
        )
    if unit.framing == "onpolicy_a1":
        if not row.a1_onpolicy:
            raise RuntimeError(f"{unit.unit_id}: row {row.conv_id} has no on-policy a1")
        return naturalistic_text_and_offsets(
            row.u1,
            row.a1_onpolicy,
            row.u2,
            assistant_tag=f"{ONPOLICY_A1_VARIANTS[unit.variant]}: ",
        )
    if unit.framing == "parent_recap":
        return parent_recap_text_and_offsets(row, tokenizer, variant=unit.variant)
    raise ValueError(f"unknown framing {unit.framing!r}")


def _validate_offsets(unit: Unit, offsets: dict[str, int], text: str) -> None:
    """Every declared slot present, in-range, and strictly ordered where the
    layout guarantees it. A degenerate (equal-boundary) pair is EXACTLY the
    realized defect this round fixes — fail loud."""
    want = set(unit.slots)
    got = set(offsets)
    # AUXILIARY offsets (e.g. `a1_start`, consumed only by the addendum-E
    # read-group builders) are allowed in the dict but are not stored slots, so
    # this is a subset check, not equality.
    if not want <= got:
        raise RuntimeError(
            f"{unit.unit_id}: slot offsets missing {sorted(want - got)} (got {sorted(got)})"
        )
    for name, off in offsets.items():
        if not (0 <= off <= len(text)):
            raise RuntimeError(f"{unit.unit_id}: slot {name} offset {off} out of range {len(text)}")
    ordered = [s for s in unit.slots]
    vals = [offsets[s] for s in ordered]
    for i in range(len(vals) - 1):
        if vals[i] >= vals[i + 1]:
            raise RuntimeError(
                f"{unit.unit_id}: non-increasing slot offsets {ordered[i]}={vals[i]} >= "
                f"{ordered[i + 1]}={vals[i + 1]} (degenerate boundary — the realized defect)"
            )


# ---------------------------------------------------------------------------
# Render driver
# ---------------------------------------------------------------------------


def render_unit(
    unit: Unit,
    staged: dict[str, Path],
    out_dir: Path,
    *,
    tokenizer_cache: dict,
    max_rows: int | None = None,
) -> dict:
    """Render one unit to ``<out_dir>/<unit_id>.jsonl``; return its manifest entry."""
    from transformers import AutoTokenizer

    if unit.model not in tokenizer_cache:
        # Load ONCE per model — a per-row from_pretrained issues a Hub
        # model_info() call each time and 429s the org quota (gotchas.md).
        tokenizer_cache[unit.model] = AutoTokenizer.from_pretrained(unit.model)
    tok = tokenizer_cache[unit.model]

    src, src_stats = load_source_rows(unit, staged)
    kept, dup, dedup_stats = dedup_rows(src)
    if max_rows is not None:
        kept, dup = kept[:max_rows], dup[:max_rows]

    out_path = out_dir / f"{unit.unit_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    dup_weight_written = 0
    dropped_len: list[dict] = []
    tok_lens: list[int] = []
    # Read-group NAMES / suffix are row-independent (they depend only on the
    # framing), so the manifest records the last written row's groups.
    groups: list[ReadGroup] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for row_idx, (row, dc) in enumerate(zip(kept, dup, strict=True)):
            text, offsets = render_row(unit, row, tok)
            _validate_offsets(unit, offsets, text)
            # Addendum E: append the deterministic turn-transition suffix (when
            # the framing needs one) and derive the X x Y read groups. The
            # SUFFIXED text is what the capture forwards.
            text, groups = build_read_groups(unit, offsets, text)
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            if n_tok > MAX_TOKENS:
                # Digest-only record: never the row text (real user content).
                dropped_len.append({"conv_id": row.conv_id, "n_tokens": n_tok})
                continue
            tok_lens.append(n_tok)
            fh.write(
                json.dumps(
                    {
                        "unit_id": unit.unit_id,
                        "row_index": row_idx,
                        "conv_id": row.conv_id,
                        "dup_count": int(dc),
                        "u2_provenance": row.u2_provenance,
                        "judge_score_mean": row.judge_score_mean,
                        "text": text,
                        "char_slots": offsets,
                        "read_groups": [asdict(g) for g in groups],
                        "n_tokens": int(n_tok),
                        "text_sha256": sha256_text(text),
                    }
                )
                + "\n"
            )
            n_written += 1
            dup_weight_written += int(dc)
    n_candidates = len(kept)
    drop_frac = len(dropped_len) / n_candidates if n_candidates else 0.0
    if drop_frac > MAX_DROP_FRACTION:
        raise RuntimeError(
            f"{unit.unit_id}: {len(dropped_len)}/{n_candidates} rows exceed MAX_TOKENS="
            f"{MAX_TOKENS} ({drop_frac:.1%} > {MAX_DROP_FRACTION:.0%}) — the token budget is "
            "systematically too small for this row distribution"
        )
    if n_written == 0:
        raise RuntimeError(f"{unit.unit_id}: zero rows written")
    entry = {
        "unit_id": unit.unit_id,
        **{k: v for k, v in asdict(unit).items()},
        "model_dir": model_dir(unit.model),
        "slots": list(unit.slots),
        "straddler_policy": {s: SLOT_STRADDLER_POLICY[s] for s in unit.slots},
        "fit_pairs": [list(p) for p in unit.fit_pairs],
        "primary_fit": PRIMARY_FIT_BY_FRAMING[unit.framing],
        # Addendum E: the X x Y grid. `read_group_names` drives the capture's
        # stored grid slots (5 per group) and the fits' 6-combo battery per group.
        "read_group_names": [g.name for g in groups],
        "grid_slots": [s for g in groups for s in g.slot_names],
        "grid_x_kinds": list(GRID_X_KINDS),
        "grid_y_kinds": list(GRID_Y_KINDS),
        "transition_suffix": groups[0].suffix_appended if groups else "",
        "rendered_path": str(out_path.relative_to(out_dir)),
        "n_rows": n_written,
        "n_dropped_over_max_tokens": len(dropped_len),
        "dropped_over_max_tokens": dropped_len[:50],
        "token_len_p50": int(sorted(tok_lens)[len(tok_lens) // 2]) if tok_lens else 0,
        "token_len_max": max(tok_lens) if tok_lens else 0,
        "dup_weight_written": dup_weight_written,
        **src_stats,
        **dedup_stats,
    }
    print(
        f"[render] {unit.unit_id}: rows={n_written} "
        f"(src={src_stats['n_source_rows']} usable={src_stats['n_usable_rows']} "
        f"dedup={dedup_stats['n_rows_after_dedup']} dropped_len={len(dropped_len)}) "
        f"tok_p50={entry['token_len_p50']} tok_max={entry['token_len_max']}",
        flush=True,
    )
    return entry


# ===========================================================================
# ADDENDA A-E (scope extension, 2026-07-30) — read groups + bridging families
# ===========================================================================
#
# Addendum E consolidates the measurement convention: every cell stores and
# fits a full X x Y grid around its ANSWER SPAN.
#
#   X_clean     last token FULLY BEFORE the answer span (straddler-EXCLUSIVE).
#               Plain text: the label's ':' — NOT the trailing space, and NOT
#               the token that fuses that space with the answer's first word.
#               Chat: the header '\n'. Story: the pre-answer open quote.
#               PRIMARY.
#   X_straddle  the token that STARTS the answer span == X_clean + 1. In plain
#               text this is the space-merged-with-first-answer-word token (the
#               PARENT's straddler-INCLUSIVE convention); in chat/story it is
#               simply the first answer token. Comparison arm — it prices what
#               the parent's convention contributed.
#   Y_mean      mean activation over the answer span (the #1345/#825 convention).
#   Y_end       last content token of the answer (the #1689 parent convention).
#   Y_boundary  the response-slot token BEFORE the next character starts
#               talking. Realized by a DETERMINISTIC, content-free turn-
#               transition suffix after the answer (chat `<|im_end|>\n
#               <|im_start|>user\n` read at the final header '\n'; plain text
#               `\n\nUser: ` read straddler-exclusive at the ':'), or by an
#               EXISTING transition already in the text (a floor-control group's
#               u1 is followed by the real assistant turn; a story answer is
#               followed by the template's own next-speaker attribution).
#
# Why straddler-EXCLUSIVE for every plain-text X (spec clarification,
# verified live): the parent capture read the naturalistic context slot with
# `straddler_include=True`, which lands ON the fused token — one token INTO
# the answer. #1345 deliberately avoided that (`issue1345_common._header_slot`
# / `_last_fully_contained`, comment "avoiding BPE straddlers"), so matching
# #1345 is what makes cross-rig comparison clean. Chat header-newline and
# story quote boundaries are atomic special/punctuation tokens and never fuse,
# so the two X variants coincide there by construction — labelled as such.

CHAT_TRANSITION_SUFFIX = "<|im_end|>\n<|im_start|>user\n"
NATURALISTIC_TRANSITION_SUFFIX = "\n\nUser: "


@dataclass(frozen=True)
class ReadGroup:
    """One answer span + the X/Y read positions around it (addendum E).

    All offsets are CHAR offsets into the unit's rendered ``text``. The capture
    rig turns each into a token index and emits five stored slots named
    ``<name>__{X_clean,X_straddle,Y_mean,Y_end,Y_boundary}``.

    ``answer_start`` is the boundary BEFORE the answer's first character;
    ``answer_end`` is the boundary AFTER its last. ``boundary_end`` is the char
    offset whose straddler-EXCLUSIVE token is the Y_boundary read (the final
    header ``\\n`` in chat, the ``:`` of ``\\n\\nUser: `` in plain text, the end
    of the story template's attribution lead-in).
    """

    name: str
    answer_start: int
    answer_end: int
    boundary_end: int
    suffix_appended: str  # "" when the transition already existed in the text

    @property
    def slot_names(self) -> tuple[str, ...]:
        return (
            f"{self.name}__X_clean",
            f"{self.name}__X_straddle",
            f"{self.name}__Y_mean",
            f"{self.name}__Y_end",
            f"{self.name}__Y_boundary",
        )


GRID_SLOT_KINDS: tuple[str, ...] = ("X_clean", "X_straddle", "Y_mean", "Y_end", "Y_boundary")
GRID_X_KINDS: tuple[str, ...] = ("X_clean", "X_straddle")
GRID_Y_KINDS: tuple[str, ...] = ("Y_mean", "Y_end", "Y_boundary")


def _chat_read_group_u2(off: dict[str, int], text: str) -> tuple[str, ReadGroup]:
    """u2-as-answer group for a chat user cell; appends the turn transition."""
    new_text = text + CHAT_TRANSITION_SUFFIX
    # Y_boundary reads at the FINAL header newline of the appended suffix.
    return new_text, ReadGroup(
        name="u2",
        answer_start=off["u2_header_end"],
        answer_end=off["u2_end"],
        boundary_end=len(new_text),
        suffix_appended=CHAT_TRANSITION_SUFFIX,
    )


def _chat_read_group_u1(off: dict[str, int], text: str) -> ReadGroup:
    """Floor-control group: u1 as the answer span.

    No suffix needed — u1's turn is ALREADY followed by the real transition
    into a1's turn, so `prev_turn_end`-style reads exist in the text. Y_boundary
    reads straddler-exclusively at the a1 header's newline, which is exactly
    ``prev_turn_end`` minus a1's content... but the simplest exact anchor is the
    offset where a1's content starts, whose preceding token is that newline.
    """
    return ReadGroup(
        name="u1",
        answer_start=off["first_user_header_end"],
        answer_end=off["u1_end"],
        # a1's content start == the end of `<|im_end|>\n<|im_start|>assistant\n`
        boundary_end=off["a1_start"],
        suffix_appended="",
    )


def chat_read_groups(off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    """Chat user cell: the u2 answer group + the u1 floor-control group."""
    new_text, g_u2 = _chat_read_group_u2(off, text)
    return new_text, [g_u2, _chat_read_group_u1(off, text)]


def naturalistic_read_groups(off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    """Naturalistic user cell: u2 answer group (suffix appended) + u1 floor."""
    new_text = text + NATURALISTIC_TRANSITION_SUFFIX
    # The `:` of the appended `\n\nUser: ` — straddler-exclusive at the offset
    # just past it, i.e. len(new_text) - 1 (the trailing space is the straddle).
    boundary = len(new_text) - 1
    g_u2 = ReadGroup(
        name="u2",
        answer_start=off["u2_header_end"],
        answer_end=off["u2_end"],
        boundary_end=boundary,
        suffix_appended=NATURALISTIC_TRANSITION_SUFFIX,
    )
    g_u1 = ReadGroup(
        name="u1",
        answer_start=off["first_user_header_end"],
        answer_end=off["u1_end"],
        # The EXISTING `\n\nAssistant: ` transition: its ':' sits one char
        # before a1's content start.
        boundary_end=off["a1_start"] - 1,
        suffix_appended="",
    )
    return new_text, [g_u2, g_u1]


def story_read_groups(off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    """Story user cell: u2 answer group; the template's own attribution
    lead-in (`". The assistant wrote back: `) IS the next-speaker transition,
    so no suffix is appended and Y_boundary reads at its end."""
    g = ReadGroup(
        name="u2",
        answer_start=off["story_prefix_end"],
        answer_end=off["u2_end"],
        boundary_end=off["parent_answer_end"],
        suffix_appended="",
    )
    return text, [g]


def single_turn_read_groups(off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    """Addendum B: the ANSWER group (suffix appended) + a u2 floor group."""
    new_text = text + NATURALISTIC_TRANSITION_SUFFIX
    g_answer = ReadGroup(
        name="answer",
        answer_start=off["answer_header_end"],
        answer_end=off["parent_answer_end"],
        # the ':' of the appended "\n\nUser: " (its trailing space straddles)
        boundary_end=len(new_text) - 1,
        suffix_appended=NATURALISTIC_TRANSITION_SUFFIX,
    )
    g_u2 = ReadGroup(
        name="u2",
        answer_start=off["first_user_header_end"],
        answer_end=off["u2_end"],
        # the ':' of the EXISTING "\n\n<Label>: " transition
        boundary_end=off["answer_header_end"] - 1,
        suffix_appended="",
    )
    return new_text, [g_answer, g_u2]


def parent_recap_read_groups(
    off: dict[str, int], text: str, *, variant: str
) -> tuple[str, list[ReadGroup]]:
    """Addendum D: the ANSWER group under the PARENT's own transition shape.

    Chat conditions append the ChatML turn transition and read Y_boundary at its
    final header newline; plain-text conditions append `\\n\\nUser: ` and read
    straddler-exclusively at its ':' — matching the base round's two shapes.
    """
    if variant.endswith("_chat"):
        new_text = text + CHAT_TRANSITION_SUFFIX
        boundary = len(new_text)
        suffix = CHAT_TRANSITION_SUFFIX
    else:
        new_text = text + NATURALISTIC_TRANSITION_SUFFIX
        boundary = len(new_text) - 1
        suffix = NATURALISTIC_TRANSITION_SUFFIX
    g = ReadGroup(
        name="answer",
        answer_start=off["answer_header_end"],
        answer_end=off["parent_answer_end"],
        boundary_end=boundary,
        suffix_appended=suffix,
    )
    return new_text, [g]


def _rg_chat(unit: Unit, off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    return chat_read_groups(off, text)


def _rg_naturalistic(unit: Unit, off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    return naturalistic_read_groups(off, text)


def _rg_story(unit: Unit, off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    return story_read_groups(off, text)


def _rg_single_turn(unit: Unit, off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    return single_turn_read_groups(off, text)


def _rg_parent_recap(unit: Unit, off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    return parent_recap_read_groups(off, text, variant=unit.variant)


READ_GROUPS_BY_FRAMING = {
    "chat": _rg_chat,
    "naturalistic": _rg_naturalistic,
    "story": _rg_story,
    # addendum C shares the naturalistic layout exactly (only the label differs)
    "onpolicy_a1": _rg_naturalistic,
    "single_turn": _rg_single_turn,
    "parent_recap": _rg_parent_recap,
}

# The read-group names each framing REALIZES, in order. Declared rather than
# only derived so (a) `_validate_read_groups` fails loud if a builder's realized
# groups ever drift from the declaration, and (b) consumers that must know the
# grid's shape WITHOUT rendering real text — the fits' synthetic smoke tree —
# read it from here instead of re-deriving it (a drifting duplicate would make
# the smoke's grid shape diverge from production's).
READ_GROUP_NAMES_BY_FRAMING: dict[str, tuple[str, ...]] = {
    "chat": ("u2", "u1"),
    "naturalistic": ("u2", "u1"),
    "story": ("u2",),
    "onpolicy_a1": ("u2", "u1"),
    "single_turn": ("answer", "u2"),
    "parent_recap": ("answer",),
}


def build_read_groups(unit: Unit, off: dict[str, int], text: str) -> tuple[str, list[ReadGroup]]:
    """Dispatch to the framing's read-group builder (addendum E).

    Returns the possibly-SUFFIXED text (the transition suffix is appended at
    render time so the capture forward covers it) plus the groups. Validated by
    :func:`_validate_read_groups`.
    """
    new_text, groups = READ_GROUPS_BY_FRAMING[unit.framing](unit, off, text)
    _validate_read_groups(unit, groups, new_text)
    return new_text, groups


def _validate_read_groups(unit: Unit, groups: list[ReadGroup], text: str) -> None:
    """Every group's four char offsets in range and strictly ordered.

    ``answer_start < answer_end <= boundary_end`` — a degenerate answer span
    (start == end) or a boundary at/inside the answer is exactly the class of
    defect this round exists to remove, so it fails loud.
    """
    if not groups:
        raise RuntimeError(f"{unit.unit_id}: no read groups")
    names = [g.name for g in groups]
    if len(set(names)) != len(names):
        raise RuntimeError(f"{unit.unit_id}: duplicate read-group names {names}")
    declared = list(READ_GROUP_NAMES_BY_FRAMING[unit.framing])
    if names != declared:
        raise RuntimeError(
            f"{unit.unit_id}: realized read groups {names} != declared "
            f"READ_GROUP_NAMES_BY_FRAMING[{unit.framing!r}] {declared} — a consumer sizing the "
            "grid off the declaration (the fits' synthetic smoke tree) would diverge"
        )
    for g in groups:
        for label, off in (
            ("answer_start", g.answer_start),
            ("answer_end", g.answer_end),
            ("boundary_end", g.boundary_end),
        ):
            if not (0 <= off <= len(text)):
                raise RuntimeError(
                    f"{unit.unit_id}/{g.name}: {label}={off} out of range {len(text)}"
                )
        if not (g.answer_start < g.answer_end <= g.boundary_end):
            raise RuntimeError(
                f"{unit.unit_id}/{g.name}: non-monotonic group offsets "
                f"start={g.answer_start} end={g.answer_end} boundary={g.boundary_end}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / ROUND_LABEL / "rendered",
        help="destination for the per-unit rendered JSONL + manifest.json",
    )
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=REPO_ROOT / "data" / "issue_1689" / ROUND_LABEL / "hf_dl",
        help="staging root for the parent source files (must resolve off /)",
    )
    ap.add_argument("--revision", default=PARENT_REVISION)
    ap.add_argument("--units", default="all", help='"all" or a comma-separated unit_id list')
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="ONE unit per FAMILY x 32 rows through the IDENTICAL code path "
        "(an explicit --units list narrows it further)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="cap rows per unit (independent of --smoke)",
    )
    ap.add_argument(
        "--gen-a1-dir",
        type=Path,
        default=None,
        help="local a1-generator output dir (same-pod path; overrides the gen_a1 Hub prefix)",
    )
    args = ap.parse_args()

    explicit_units = args.units != "all"
    units = (
        list(UNITS)
        if not explicit_units
        else [UNIT_BY_ID[u.strip()] for u in args.units.split(",") if u.strip()]
    )
    max_rows = args.max_rows
    if args.smoke:
        # The smoke covers ONE cell per FAMILY, not one overall: every family has
        # its own offset builder, source loader and read-group shape, so a
        # single-family smoke is blind to the others' seams (the per-arm-class
        # smoke duty). An explicit --units list is HONORED under --smoke so a
        # per-family smoke can be run on its own.
        if not explicit_units:
            units = smoke_units()
        max_rows = max_rows or 32

    staged = stage_source_files(args.stage_root, revision=args.revision, gen_a1_dir=args.gen_a1_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_cache: dict = {}
    entries = [
        render_unit(u, staged, args.out_dir, tokenizer_cache=tokenizer_cache, max_rows=max_rows)
        for u in units
    ]
    manifest = {
        "metadata": base_metadata(),
        "smoke": bool(args.smoke),
        "max_tokens": MAX_TOKENS,
        "lmsys_const_u2_sha256": sha256_text(LMSYS_CONST_U2),
        "realized_defects_fixed": {
            "naturalistic_boundary_collapse": "prefix_end == context_end == answer_end in the "
            "parent user naturalistic cells (identity+bias R2 = 1.0 exactly)",
            "stale_u2_in_prompt": "parent fill phases updated u2_text but not "
            "messages[-1].content / u2_text_marked / prompt_text, so haiku + on-policy "
            "user cells embedded the 25-char <UNFILLED_U2_PLACEHOLDER>",
            "lmsys_is_constant_u2": "the two-turn corpus has no u2_lmsys field; the parent "
            "lmsys arm is a CONSTANT 34-char fallback u2, not LMSYS-sourced (NOT fixable "
            "from persisted data — carried forward labelled u2_provenance=const_fallback)",
        },
        "units": entries,
    }
    man_path = args.out_dir / "manifest.json"
    with man_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    total = sum(e["n_rows"] for e in entries)
    print(f"[render] wrote {len(entries)} units / {total} rows -> {man_path}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    # Explicit exit BEFORE C-extension finalization (transformers/tokenizers):
    # a PyGILState_Release atexit race can otherwise rewrite the return code and
    # abort a `set -e` dispatcher on a COMPLETED phase (gotchas.md).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
