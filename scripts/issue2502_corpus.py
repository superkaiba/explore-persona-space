#!/usr/bin/env python
"""Issue #2502 — P0 corpus build: 150k maximally-diverse context corpus.

Pure data engineering (stream -> per-source filter -> two-stage dedup ->
budget-scale -> split + LODO folds -> write JSONL). NO model inference, NO
pod provisioning. Follows plan v6 (`tasks/running/2502/plans/plan.md`) §4 P0.

Reuses the #1739 streaming/dedup machinery verbatim
(`src/explore_persona_space/experiments/issue_1739/corpus_staging.py`):
``_stream_stage`` (per-chunk checkpoint + fingerprint-gated resume, per-filter
reject counters, fail-loud on kept==0), ``_hf_stream`` (HF ``streaming=True``,
never a full download), ``minhash_signatures`` + the 16x4 LSH banding of
``near_dup_mask`` (the CANDIDATE stage), ``usable_text`` / ``norm_text`` /
``first_human_turn`` / ``_subsample`` / ``_fingerprint`` / ``read_jsonl`` /
``_write_jsonl_atomic``.

MF-K (binding): dedup is TWO stages on the CONTEXT text, WITHIN and ACROSS all
sources, BEFORE any train/val/test assignment — (a) the LSH CANDIDATE stage
(16x4 banding over 64 MinHash perms, flags Jaccard ~>=0.5) THEN (b) the exact
char-5-gram Jaccard >= 0.8 CONFIRM on candidates (the full #1775 criterion;
reproduces ``FC.exact_jaccard`` semantics from
``scripts/issue1775_fold_check.py:159`` — ``NEAR_DUPE_NGRAM=5``,
``NEAR_DUPE_JACCARD=0.8``, ``MINHASH_CAND_EST=0.6`` — WITHOUT the heavy
torch/vLLM import chain that module carries; see the reimplemented
``_char_ngrams`` / ``_exact_jaccard`` below). ``near_dup_mask`` alone (candidate
~0.5) would over-delete; the exact-Jaccard confirm is load-bearing. Fail loud
if any source's post-dedup kept count is 0.

Family 12 (plan §4 item 12) has TWO parts, both implemented here:
  (a) Persona/system-prompt PREFIXES — ``proj-persona/PersonaHub`` (config
      ``persona``) + ``nvidia/Nemotron-Personas-USA`` + ``fka/prompts.chat``
      CROSSED with the fixed in-repo query bank ``wildchat_random_v1``
      (``artifacts.banks.load_bank``; 600 toxic/redacted-screened real user
      queries): composed context = ``<prefix>\\n\\n<query>``, seeded pairing
      (the #1739 ``build_evil_cross`` pattern) — family budget ~4k;
  (b) Domain fillers (Magicoder / NuminaMath / PubMedQA / ChatDoctor /
      legalbench) marked ``topup=True``: they absorb the REMAINING budget to
      exactly 150k after every non-filler source takes its realized yield
      (``allocate_with_topup``); the big three carry generous staging caps so
      the 150k target is structurally reachable.

RESUME SOUNDNESS: every stage fingerprint is built by ONE shared builder
(``_stage_fingerprint``) carrying every output-affecting key — the pinned
upstream dataset REVISION (resolved once per dataset, also threaded into the
stream so a resumed ``skip_scanned`` fast-forward reads the SAME bytes the
checkpoint scanned), keep_cap, filters version, token budget + tokenizer ids,
language filter, stream_cap, and the fallback flag — for the PRIMARY and the
FALLBACK path alike. The within-source ``seen`` dedup set is re-seeded from
matching partial checkpoints and from completed configs' pools, so a resume
never re-admits duplicates of pre-crash kept rows.

CROSS-MODEL TOKEN FILTER: the MAX-context-token filter rejects a row when its
tokenized length exceeds the budget under EITHER model tokenizer (Qwen2.5-7B
-Instruct AND Qwen3.5-9B), so both model arms consume the IDENTICAL corpus and
gen_capture never needs a per-model drop.

ROUND-10 SCHEMA GATE (anti-whack-a-mole, schema level): per-source FIELD
schemas are verified by ``verify_source_schemas`` BEFORE any staging — a
bounded tiny-real streaming probe of EVERY (source, config, split[, file])
attempt through the FULL production keep chain, aggregating every cast-crash /
kept==0 / stream defect into ONE raise (the round-9 ``verify_declared_splits``
generalized from split METADATA to actual schemas). Multi-file
schema-inferring sources are probed PER FILE with cross-file signature
comparison (datasets infers features from the first file and casts later
files to them — the sycophancy_eval `_cast_table` crash class), and
data_files sources STAGE per file (independent inference; no cross-file cast
exists to crash).

CONTENT HYGIENE (binding, `.claude/rules/trigger-dense-review.md`): several
corpora are harmful-content / real-user / jailbreak banks. This module NEVER
logs, prints, or persists item TEXT through any counts/report channel — the
corpus JSONL on disk carries the text; every log line, the dedup report, and
the ``--probe`` output carry counts, ids (sha digests), source tags, regime
classes, and field NAMES only.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.experiments.issue_1739 import corpus_staging as CS
from explore_persona_space.orchestrate import hub
from explore_persona_space.orchestrate.env import load_dotenv

logger = logging.getLogger("issue2502.corpus")

# --- recipe pins ------------------------------------------------------------
BUDGET = 150_000
SEED = 42
# Length filter (plan §4): MIN from #1739; MAX = a context-token budget so
# every kept context + the 1024 gen cap + the 2048 re-gen headroom fits
# max_model_len=8192 on BOTH engines. BOTH model tokenizers filter (a kept row
# fits Model A AND Model B — the cross-model common-corpus contract).
MIN_TEXT_CHARS = CS.MIN_TEXT_CHARS  # 16
MAX_MODEL_LEN = 8192
GEN_HEADROOM_TOKENS = 2048  # re-gen headroom the token budget must reserve
TOKEN_MARGIN = 512  # chat-template + generation-prompt overhead margin
DEFAULT_TOKENIZERS = ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3.5-9B")
# Fingerprint recipe versions (bumping either forces a restage of stale pools).
KEEP_FILTERS_VERSION = "issue2502.keep.v2"  # v2: dual-tokenizer filter + pinned revision
PREFIX_FILTERS_VERSION = "issue2502.prefix.v1"  # family-12 prefix staging (pre-cross)
# Family-12 crossing (plan §4 item 12): fixed in-repo query bank + join.
CROSS_QUERY_BANK = "wildchat_random"  # artifacts.banks name; 600 screened real queries
CROSS_JOIN = "\n\n"  # composed context = prefix + CROSS_JOIN + query
# Dedup (MF-K): the exact-Jaccard CONFIRM stage on top of the LSH candidates.
NEAR_DUPE_NGRAM = 5  # char n-gram width (== issue1775_fold_check NEAR_DUPE_NGRAM)
NEAR_DUPE_JACCARD = 0.8  # confirm threshold (== issue1775_fold_check NEAR_DUPE_JACCARD)
MINHASH_CAND_EST = 0.6  # documented LSH candidate floor (16x4 banding S-curve)

# Split (plan §11): per-source stratified holdout.
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Regime classes (committed source->regime table, plan §5, MF-I).
REGIME_ORDINARY = "ordinary"
REGIME_WEIRD = "weird"
REGIME_NEAR = "near-distribution"
REGIME_IDIO = "idiosyncratic"
REGIME_CLASSES = (REGIME_ORDINARY, REGIME_WEIRD, REGIME_NEAR, REGIME_IDIO)


# ---------------------------------------------------------------------------
# Source registry — 12 families (plan §4), per-source keep predicate + PRE-dedup
# keep-cap + regime_class + realism_tier. Field/config schemas for the real
# sources are `probe-verified`: the counts-only ``--probe`` (run by the
# orchestrator pre-launch, per-filter reject counters + fail-loud-on-zero) is
# the schema-from-artifact gate — the harmful banks cannot be row-read on the
# VM (content hygiene). The corpus_staging-verified schemas (hh-rlhf transcript,
# toxic-chat, in-the-wild-jailbreak `prompt`) are inherited directly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One corpus source substratum (== one LODO group == one ``source_tag``)."""

    source_tag: str
    dataset_id: str
    regime_class: str
    realism_tier: int
    pre_dedup_cap: int
    configs: tuple[str | None, ...] = (None,)
    # Split(s) to stage PER CONFIG (attempts = configs x splits; the keep-cap
    # stays cumulative across attempts). Round-9 crash class: the registry
    # default 'train' is NOT universal — JBB-Behaviors' `behaviors` config
    # offers ['harmful','benign'] only, and inheriting 'train' crashed the
    # live P0 build (ValueError: Bad split). verify_declared_splits() checks
    # every declared (config, split) against the dataset BEFORE staging.
    splits: tuple[str, ...] = ("train",)
    data_dir: str | None = None
    # Pin + load from this repo ref instead of the default branch (e.g.
    # 'refs/convert/parquet' for legacy script datasets — datasets>=3 refuses
    # loading scripts, and the auto-converted parquet branch is the only
    # loadable form). The ref's resolved sha enters the fingerprint + stream
    # exactly like a main-branch sha.
    revision_ref: str | None = None
    # Stream raw repo FILES through the packaged `json` builder instead of the
    # dataset id: hf:// data_files template(s) whose '{revision}' is
    # substituted with the pinned sha. TWO use cases: (a) PIPPA-class repos
    # with a dead script but raw jsonl (single template); (b) round-10:
    # multi-file repos whose files carry HETEROGENEOUS schemas — the combined
    # load infers features from the FIRST file and datasets' `_cast_table`
    # crashes at the file boundary (sycophancy_eval: feedback.jsonl's
    # `metadata` struct adds `prompt_template_type` at row 12,157), so EACH
    # template is staged as its OWN attempt with independently-inferred
    # features (no cross-file cast ever happens). Split-fixed by construction:
    # a str data_files serves exactly the single 'train' split.
    data_files_template: str | tuple[str, ...] | None = None
    # Ordered candidate SCALAR text fields (first present usable one wins).
    text_fields: tuple[str, ...] = ()
    # Ordered candidate CONVERSATION fields (message lists -> first user turn).
    conv_fields: tuple[str, ...] = ()
    # Use CS.first_human_turn on a raw transcript STRING field (hh-rlhf shape).
    transcript_field: str | None = None
    # Round 10: conv extraction accepts the FIRST turn REGARDLESS of speaker
    # (OpenCAI-class roleplay logs: turns carry {'author': <character name>,
    # 'message': str} — no user/assistant designation exists, and the opener
    # IS the roleplay context; probe-verified 2026-08-25).
    conv_any_speaker: bool = False
    # Keep only English rows when this source carries a `language` field (full
    # name, e.g. 'English' — NOT an ISO code; gotchas real-corpus streaming).
    filter_language: bool = False
    # Route rows to `{source_tag}__flagged` (near-distribution) when the
    # moderation flag is set, else `{source_tag}__ordinary`/`{regime_class}`.
    moderation_split: bool = False
    gated: bool = False
    # Optional fallback dataset id when the primary is a gated read failure.
    fallback_dataset_id: str | None = None
    # Family-12 crossing (plan §4 item 12): when set, staged rows are
    # persona/system-prompt PREFIXES crossed with this fixed in-repo query
    # bank (artifacts.banks name); composed context = prefix + "\n\n" + query.
    cross_query_bank: str | None = None
    # Domain-filler top-up lever (plan §4 item 12: "remaining budget to
    # 150k"): topup sources absorb the residual budget AFTER every non-topup
    # source takes its realized yield (allocate_with_topup).
    topup: bool = False


def _data_files_list(spec: SourceSpec) -> tuple[str, ...]:
    """Normalized per-file data_files templates ((), one, or many)."""
    if spec.data_files_template is None:
        return ()
    if isinstance(spec.data_files_template, str):
        return (spec.data_files_template,)
    return tuple(spec.data_files_template)


def _file_stem(template: str) -> str:
    """Basename stem of an hf:// data_files template (attempt key / out-path
    part; repo METADATA, content-safe)."""
    return template.rsplit("/", 1)[-1].split(".", 1)[0]


def _probe_file_format(url: str) -> str | None:
    """Packaged-builder name for a data file URL, or None (not stageable
    per-file / not a schema-INFERRING format — parquet carries an explicit
    schema and needs no per-file treatment)."""
    name = url.split("?")[0].rsplit("/", 1)[-1].lower()
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    return {"json": "json", "jsonl": "json", "csv": "csv", "tsv": "csv"}.get(ext)


def _iter_attempts(spec: SourceSpec) -> tuple[tuple[str | None, str, str | None], ...]:
    """The (config, split, data_file_template) staging attempts of a source.

    data_files sources stage ONE attempt PER FILE (independent feature
    inference — the round-10 cross-file cast fix); everything else keeps the
    round-9 configs x splits grid. Raises on unsupported combinations so a
    registry edit fails at import/selfcheck, never mid-stream.
    """
    files = _data_files_list(spec)
    if not files:
        return tuple((c, s, None) for c in spec.configs for s in spec.splits)
    if spec.fallback_dataset_id is not None or spec.cross_query_bank is not None:
        raise ValueError(
            f"{spec.source_tag}: data_files sources support neither fallback_dataset_id "
            "nor cross_query_bank"
        )
    bad = [f for f in files if _probe_file_format(f) is None]
    if bad:
        raise ValueError(f"{spec.source_tag}: unstageable data_files format(s): {bad}")
    return tuple((spec.configs[0], "train", f) for f in files)


def _regime_tags(spec: SourceSpec) -> dict[str, str]:
    """The source_tag -> regime_class entries this spec contributes."""
    if spec.moderation_split:
        return {
            f"{spec.source_tag}__ordinary": spec.regime_class,
            f"{spec.source_tag}__flagged": REGIME_NEAR,
        }
    return {spec.source_tag: spec.regime_class}


# The 12 families. Combined family budgets are split across substrata; the P0
# ``--probe`` re-scales the 150k budget against realized per-source post-dedup
# yields, so these PRE-dedup caps are deliberately generous (their sum exceeds
# 150k).
SOURCES: tuple[SourceSpec, ...] = (
    # 1. WildChat (tier-1 real user chat; moderation-flag regime split).
    SourceSpec(
        source_tag="wildchat",
        dataset_id="allenai/WildChat-1M-Full",
        fallback_dataset_id="allenai/WildChat-4.8M",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=30_000,
        conv_fields=("conversation",),
        filter_language=True,
        moderation_split=True,
        gated=True,
    ),
    # 2. LMSYS-Chat-1M (tier-1 real user chat; moderation-flag regime split).
    SourceSpec(
        source_tag="lmsys_chat_1m",
        dataset_id="lmsys/lmsys-chat-1m",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=25_000,
        conv_fields=("conversation",),
        filter_language=True,
        moderation_split=True,
        gated=True,
    ),
    # 3. in-the-wild jailbreak prompts — jailbreak configs (weird) + regular
    #    configs (ordinary, ingested to reach a meaningful yield; FC-2).
    SourceSpec(
        source_tag="itw_jailbreak",
        dataset_id="TrustAIRLab/in-the-wild-jailbreak-prompts",
        regime_class=REGIME_WEIRD,
        realism_tier=1,
        pre_dedup_cap=8_000,
        configs=("jailbreak_2023_12_25", "jailbreak_2023_05_07"),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="itw_regular",
        dataset_id="TrustAIRLab/in-the-wild-jailbreak-prompts",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=7_000,
        configs=("regular_2023_12_25", "regular_2023_05_07"),
        text_fields=("prompt",),
    ),
    # 4. hh-rlhf red-team attempts (first human turn; weird).
    SourceSpec(
        source_tag="hh_redteam",
        dataset_id="Anthropic/hh-rlhf",
        regime_class=REGIME_WEIRD,
        realism_tier=1,
        pre_dedup_cap=12_000,
        data_dir="red-team-attempts",
        transcript_field="transcript",
    ),
    # 5. adversarial + matched-benign twins (weird).
    SourceSpec(
        source_tag="wildjailbreak",
        dataset_id="allenai/wildjailbreak",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=4_000,
        configs=("train",),
        text_fields=("adversarial", "vanilla", "prompt"),
        gated=True,
    ),
    SourceSpec(
        source_tag="jbb_behaviors",
        dataset_id="JailbreakBench/JBB-Behaviors",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        configs=("behaviors",),
        # The `behaviors` config offers NO 'train' split — its splits are
        # ['harmful','benign'] (probe-verified 2026-08-24; the round-9 live
        # crash). BOTH are staged under this one weird substratum: the plan
        # §4 item-5 family IS "adversarial + matched-benign twins" (the
        # wildjailbreak adversarial+vanilla pattern), and JBB's benign split
        # is the paired benign twin of each harmful behavior — with gated
        # wildjailbreak skipped it is the family's only realized twin set.
        splits=("harmful", "benign"),
        text_fields=("Goal", "goal", "prompt"),
    ),
    SourceSpec(
        source_tag="advbench",
        dataset_id="walledai/AdvBench",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("prompt", "goal"),
    ),
    SourceSpec(
        source_tag="harmbench",
        dataset_id="walledai/HarmBench",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        configs=("standard",),
        text_fields=("prompt", "behavior"),
    ),
    # 6. sycophancy / deception / trait (near-distribution).
    SourceSpec(
        source_tag="model_written_evals",
        dataset_id="Anthropic/model-written-evals",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=4_000,
        # No 'sycophancy' BuilderConfig exists (only 'default'; probe-verified
        # 2026-08-24) — the sycophancy scoping is the repo DIRECTORY layout.
        # Round-10 schema-gate catch: the dir's jsonl files have heterogeneous
        # top-level fields (political_typology_quiz adds `user_affiliation`) —
        # a LATENT `_cast_table` crash the round-9 build missed only because
        # the 4k keep-cap stopped the stream inside file 1 — so each file
        # stages as its OWN attempt (independent feature inference), replacing
        # the data_dir route. Field 'question' in all (probe-verified
        # 2026-08-25). sycophancy_on_philpapers2020.jsonl is EXCLUDED: it is
        # BYTE-IDENTICAL upstream to sycophancy_on_nlp_survey.jsonl (same file
        # sha256, 9,984 rows, 100% line overlap — raw-file probe 2026-08-25),
        # so staging it keeps 0 rows after within-source dedup and would trip
        # the kept==0 fail-loud at production scale.
        data_files_template=(
            "hf://datasets/Anthropic/model-written-evals@{revision}/sycophancy/"
            "sycophancy_on_nlp_survey.jsonl",
            "hf://datasets/Anthropic/model-written-evals@{revision}/sycophancy/"
            "sycophancy_on_political_typology_quiz.jsonl",
        ),
        text_fields=("question", "statement", "prompt"),
    ),
    SourceSpec(
        source_tag="sycophancy_eval",
        dataset_id="meg-tong/sycophancy-eval",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=3_000,
        # Round-10 live-crash fix (probe-verified 2026-08-25, structural probe
        # of revision 18f18160): the repo is 4 heterogeneous jsonl files whose
        # `metadata` structs differ (feedback.jsonl adds prompt_template_type
        # -> datasets `_cast_table` TypeError at row 12,157, the file
        # boundary), so each file stages as its OWN attempt with independent
        # feature inference. Context text lives in `prompt` = a list of
        # {"type": "human"|"ai", "content": str} turns (NOT a scalar field —
        # the round-9 text_fields kept 0 of 12,000 scanned rows);
        # first_user_content handles the `type` role key.
        data_files_template=(
            "hf://datasets/meg-tong/sycophancy-eval@{revision}/answer.jsonl",
            "hf://datasets/meg-tong/sycophancy-eval@{revision}/are_you_sure.jsonl",
            "hf://datasets/meg-tong/sycophancy-eval@{revision}/feedback.jsonl",
            "hf://datasets/meg-tong/sycophancy-eval@{revision}/mimicry.jsonl",
        ),
        conv_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="mask",
        dataset_id="cais/MASK",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=3_000,
        # cais/MASK REQUIRES a config ("Config name is missing" with None)
        # and every config offers only a 'test' split (probe-verified
        # 2026-08-24, all six). All six scenario configs staged for
        # deception-scenario diversity (plan §4 family 6).
        configs=(
            "continuations",
            "disinformation",
            "doubling_down_known_facts",
            "known_facts",
            "provided_facts",
            "statistics",
        ),
        splits=("test",),
        text_fields=("user_prompt", "prompt", "question"),
    ),
    # 7. over-refusal boundary (near-distribution).
    SourceSpec(
        source_tag="or_bench",
        dataset_id="bench-llm/or-bench",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=4_000,
        configs=("or-bench-80k",),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="coconot",
        dataset_id="allenai/coconot",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=2_500,
        configs=("original",),
        text_fields=("prompt", "question"),
    ),
    SourceSpec(
        source_tag="xstest",
        dataset_id="walledai/XSTest",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=1_500,
        # walledai/XSTest offers ONLY a 'test' split (probe-verified
        # 2026-08-24) — the registry-default 'train' would crash.
        splits=("test",),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="do_not_answer",
        dataset_id="LibrAI/do-not-answer",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=2_000,
        text_fields=("question", "prompt"),
    ),
    # 8. roleplay-as-someone (idiosyncratic).
    SourceSpec(
        source_tag="pippa",
        dataset_id="PygmalionAI/PIPPA",
        regime_class=REGIME_IDIO,
        realism_tier=1,
        pre_dedup_cap=5_000,
        # PIPPA is a legacy SCRIPT dataset ("Dataset scripts are no longer
        # supported, but found PIPPA.py" under datasets>=3; no
        # refs/convert/parquet branch exists — probe-verified 2026-08-24).
        # The repo still carries the raw jsonl, so rows stream through the
        # packaged json builder off the pinned revision. pippa_deduped.jsonl
        # is the repo's filtered/deduped variant — the plan §4 item-8
        # "PIPPA (filtered)" pin. Conversation turns use the
        # {'message','is_human'} shape, which first_user_content handles.
        data_files_template=("hf://datasets/PygmalionAI/PIPPA@{revision}/pippa_deduped.jsonl"),
        conv_fields=("conversation",),
    ),
    SourceSpec(
        source_tag="opencai",
        dataset_id="Norquinal/OpenCAI",
        regime_class=REGIME_IDIO,
        realism_tier=1,
        pre_dedup_cap=3_000,
        # Round-10 schema-gate catch (kept=0/150): OpenCAI turns are
        # {'author': <character name>, 'message': str} Discord roleplay logs —
        # NO user/assistant designation exists (probe-verified 2026-08-25),
        # so extraction takes the FIRST turn of any speaker (the roleplay
        # opener IS the context). The old text_fields ('text','input') match
        # nothing (real top-level fields: conversations/timestamp/type/
        # token-length stats).
        conv_fields=("conversations", "conversation"),
        conv_any_speaker=True,
    ),
    # 9. surreal / constrained-form / cipher-ASCII (idiosyncratic).
    SourceSpec(
        source_tag="writingprompts",
        dataset_id="euclaise/writingprompts",
        regime_class=REGIME_IDIO,
        realism_tier=1,
        pre_dedup_cap=3_000,
        text_fields=("prompt", "text"),
    ),
    SourceSpec(
        source_tag="opus_writingprompts",
        dataset_id="Gryphe/Opus-WritingPrompts",
        regime_class=REGIME_IDIO,
        realism_tier=3,
        pre_dedup_cap=2_000,
        conv_fields=("conversations",),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="ifeval",
        dataset_id="google/IFEval",
        regime_class=REGIME_IDIO,
        realism_tier=2,
        pre_dedup_cap=800,
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="riddle_sense",
        dataset_id="INK-USC/riddle_sense",
        regime_class=REGIME_IDIO,
        realism_tier=3,  # plan §4 names it in the tier-3/tier-4 minority (g1 m4)
        pre_dedup_cap=1_500,
        # Legacy SCRIPT dataset (unloadable under datasets>=3); its
        # auto-converted parquet branch IS loadable — splits
        # ['train','validation','test'], plain-string 'question' field
        # (probe-verified 2026-08-24, streamed row 1 field names).
        revision_ref="refs/convert/parquet",
        text_fields=("question",),
    ),
    SourceSpec(
        source_tag="retro_ascii_art",
        dataset_id="jdpressman/retro-ascii-art-v1",
        regime_class=REGIME_IDIO,
        realism_tier=4,
        pre_dedup_cap=1_500,
        text_fields=("prompt", "instruction", "text"),
    ),
    # 10. harm-category / trust (weird).
    SourceSpec(
        source_tag="beavertails",
        dataset_id="PKU-Alignment/BeaverTails",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=4_000,
        # BeaverTails has NO '30k_train' BuilderConfig — only 'default'
        # (probe-verified 2026-08-24); the 30k/330k families are SPLITS
        # ['330k_train','330k_test','30k_train','30k_test'], field 'prompt'.
        splits=("30k_train",),
        text_fields=("prompt",),
    ),
    SourceSpec(
        source_tag="aegis",
        dataset_id="nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=3_000,
        text_fields=("prompt", "text"),
    ),
    SourceSpec(
        source_tag="decodingtrust",
        dataset_id="AI-Secure/DecodingTrust",
        regime_class=REGIME_WEIRD,
        realism_tier=2,
        pre_dedup_cap=2_000,
        text_fields=("prompt", "text"),
        gated=True,
    ),
    # 11. tulu-3 SFT mixture (broad-instruction balance; ordinary).
    SourceSpec(
        source_tag="tulu3",
        dataset_id="allenai/tulu-3-sft-mixture",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=8_000,
        conv_fields=("messages",),
    ),
    # 12a. persona/system-prompt PREFIXES crossed with the fixed query bank
    # (plan §4 item 12: PersonaHub + Nemotron-Personas-USA + prompts.chat
    # crossed with a fixed query bank — family budget ~4k; per-source caps sum
    # to 4k). Schemas probe-verified 2026-08-23 via datasets-server /info:
    # PersonaHub config `persona` field `persona` (200k rows); Nemotron
    # default field `persona` (1M rows); prompts.chat default field `prompt`
    # (2,134 rows).
    SourceSpec(
        source_tag="personahub_prefix",
        dataset_id="proj-persona/PersonaHub",
        regime_class=REGIME_NEAR,
        realism_tier=3,
        pre_dedup_cap=1_500,
        configs=("persona",),
        text_fields=("persona",),
        cross_query_bank=CROSS_QUERY_BANK,
    ),
    SourceSpec(
        source_tag="nemotron_prefix",
        dataset_id="nvidia/Nemotron-Personas-USA",
        regime_class=REGIME_NEAR,
        realism_tier=3,
        pre_dedup_cap=1_500,
        text_fields=("persona",),
        cross_query_bank=CROSS_QUERY_BANK,
    ),
    SourceSpec(
        source_tag="promptschat_prefix",
        dataset_id="fka/prompts.chat",
        regime_class=REGIME_NEAR,
        realism_tier=2,
        pre_dedup_cap=1_000,
        text_fields=("prompt",),
        cross_query_bank=CROSS_QUERY_BANK,
    ),
    # 12b. domain fillers (in-distribution corners) — the plan's "remaining
    # budget to 150k" TOP-UP lever: topup=True + generous staging caps on the
    # three large sources (Magicoder 75k / NuminaMath 860k / ChatDoctor 100k
    # upstream rows), so under-realized weird/near sources cannot strand the
    # corpus below 150k. pubmedqa (pqa_labeled: 1k rows) + legalbench
    # (hearsay: ~95 rows) are yield-bound and cannot top up; caps stay modest.
    SourceSpec(
        source_tag="magicoder",
        dataset_id="ise-uiuc/Magicoder-OSS-Instruct-75K",
        regime_class=REGIME_ORDINARY,
        realism_tier=3,  # LLM-generated synthetic corpus (data-realism tier 3; g1 m4)
        pre_dedup_cap=15_000,
        text_fields=("problem", "instruction"),
        topup=True,
    ),
    SourceSpec(
        source_tag="numinamath",
        dataset_id="AI-MO/NuminaMath-CoT",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=15_000,
        text_fields=("problem", "question"),
        topup=True,
    ),
    SourceSpec(
        source_tag="pubmedqa",
        dataset_id="qiaojin/PubMedQA",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=2_000,
        configs=("pqa_labeled",),
        text_fields=("question",),
        topup=True,
    ),
    SourceSpec(
        source_tag="chatdoctor",
        dataset_id="lavita/ChatDoctor-HealthCareMagic-100k",
        regime_class=REGIME_ORDINARY,
        realism_tier=1,
        pre_dedup_cap=15_000,
        text_fields=("input", "instruction"),
        topup=True,
    ),
    SourceSpec(
        source_tag="legalbench",
        dataset_id="nguha/legalbench",
        regime_class=REGIME_ORDINARY,
        realism_tier=2,
        pre_dedup_cap=2_000,
        configs=("hearsay",),
        # hearsay splits are train=5 (few-shot exemplars) + test=94
        # (probe-verified 2026-08-24); 'train' alone would stage ~5 rows vs
        # the plan's "~95". Both splits staged — same distribution, and the
        # source is yield-bound (plan §4 item 12).
        splits=("train", "test"),
        text_fields=("text", "question"),
        topup=True,
    ),
)


def build_source_regime_table(sources: tuple[SourceSpec, ...] = SOURCES) -> dict[str, str]:
    """Committed source_tag -> regime_class map (MF-I; every REALIZED tag must
    resolve here, else evaluation refuses an unknown tag)."""
    table: dict[str, str] = {}
    for spec in sources:
        for tag, regime in _regime_tags(spec).items():
            if regime not in REGIME_CLASSES:
                raise ValueError(f"source {spec.source_tag}: bad regime {regime!r}")
            table[tag] = regime
    return table


# ---------------------------------------------------------------------------
# Text extraction (defensive; harmful-content-safe — never returns via a log).
# ---------------------------------------------------------------------------


def first_user_content(conv: object) -> str | None:
    """First user/human turn content of a message-list conversation.

    Role key: ``role`` (WildChat/LMSYS/tulu) -> ``from`` (OpenCAI/Opus) ->
    ``type`` (sycophancy-eval's {"type": "human"|"ai", "content"} shape —
    round 10) -> the PIPPA ``is_human`` bool. The ``type`` addition is
    strictly additive AFTER role/from (no already-staged source's keep
    decisions change — their turns carry role/from — so
    KEEP_FILTERS_VERSION deliberately does NOT bump; resume-compat, round 9).
    """
    if not isinstance(conv, list):
        return None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or turn.get("from") or turn.get("type") or "").lower()
        is_human = turn.get("is_human") is True
        if role in ("user", "human") or is_human:
            content = turn.get("content") or turn.get("value") or turn.get("message")
            if isinstance(content, str) and content.strip():
                return content
    return None


def first_turn_content(conv: object) -> str | None:
    """First turn's content of a message-list conversation REGARDLESS of
    speaker (round 10, ``conv_any_speaker`` sources: OpenCAI-class roleplay
    logs have no user/assistant designation — the opener is the context)."""
    if not isinstance(conv, list):
        return None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        content = turn.get("content") or turn.get("value") or turn.get("message")
        if isinstance(content, str) and content.strip():
            return content
    return None


def extract_context(raw: dict, spec: SourceSpec) -> str | None:
    """Extract the single context text for a raw row per its SourceSpec.

    Priority: transcript (first human turn) -> scalar text_fields ->
    conversation fields (first user turn; first turn of ANY speaker for
    ``conv_any_speaker`` sources). Returns None when nothing usable is
    present (the row is then counted as a per-filter reject, never crashed).
    """
    if spec.transcript_field is not None:
        return CS.first_human_turn(raw.get(spec.transcript_field))
    for f in spec.text_fields:
        v = raw.get(f)
        if isinstance(v, str) and v.strip():
            return v
    conv_extract = first_turn_content if spec.conv_any_speaker else first_user_content
    for cf in spec.conv_fields:
        content = conv_extract(raw.get(cf))
        if content is not None:
            return content
    return None


def _is_moderation_flagged(raw: dict) -> bool:
    """Moderation-flag detector for WildChat/LMSYS rows (plan §5: the
    ORDINARY vs NEAR-DISTRIBUTION split of families 1-2 is on the MODERATION
    flag).

    Keys ONLY on moderation verdicts — the top-level `toxic` bool and the
    per-turn-aligned `openai_moderation` list-of-dicts' `flagged` bool.
    `redacted` is PII-masking, NOT a moderation verdict, and deliberately does
    NOT flag (g1 m5: PII-redacted-but-benign rows must stay ordinary).
    Undetectable -> non-flagged (ordinary); the ``--probe`` per-regime counts
    surface a field-name miss (flagged==0).
    """
    if raw.get("toxic") is not None and CS.parse_bool_field(raw.get("toxic")):
        return True
    mod = raw.get("openai_moderation")
    if isinstance(mod, list):
        for entry in mod:
            if isinstance(entry, dict) and entry.get("flagged") is True:
                return True
    return False


def _row_language(raw: dict) -> str | None:
    v = raw.get("language")
    return v if isinstance(v, str) else None


# ---------------------------------------------------------------------------
# Per-source token-length filter (plan §4 MAX-context-token filter).
# ---------------------------------------------------------------------------


class TokenLengthFilter:
    """Reject contexts whose tokenized length exceeds the context-token budget
    under EITHER model tokenizer (the cross-model common-corpus contract).

    P0 filters against BOTH Qwen2.5-7B-Instruct AND Qwen3.5-9B (tokenizer_class
    Qwen2Tokenizer — loads under the repo-standard transformers), keeping only
    rows that fit BOTH budgets, so the two model arms consume the IDENTICAL
    corpus and gen_capture's per-model token-budget drop can never fire
    (codex round-1 BLOCKER cross-model-token-filter). Budget = max_model_len -
    gen_headroom - margin (raw context text is a single user turn; the
    chat-template + generation-prompt overhead lands inside the margin, so the
    budget is conservative for both engines). Lazy tokenizer load — never
    imported by ``--import-check`` or the synthetic ``--selfcheck``, which
    injects fake ``_count_fns`` instead.
    """

    def __init__(self, tokenizer_ids: tuple[str, ...] | list[str], budget_tokens: int):
        self.tokenizer_ids = tuple(tokenizer_ids)
        if not self.tokenizer_ids:
            raise ValueError("TokenLengthFilter needs >= 1 tokenizer id")
        self.budget_tokens = int(budget_tokens)
        # Injectable seam (selfcheck): list of text -> token-count callables.
        self._count_fns: list[Callable[[str], int]] | None = None

    def _counters(self) -> list[Callable[[str], int]]:
        if self._count_fns is None:
            from transformers import AutoTokenizer

            def _mk(tok) -> Callable[[str], int]:
                return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])

            self._count_fns = [
                _mk(AutoTokenizer.from_pretrained(tid)) for tid in self.tokenizer_ids
            ]
        return self._count_fns

    def too_long(self, text: str) -> bool:
        """True when the text exceeds the budget under ANY model tokenizer."""
        return any(fn(text) > self.budget_tokens for fn in self._counters())


def token_budget(max_model_len: int, gen_headroom: int, margin: int) -> int:
    budget = max_model_len - gen_headroom - margin
    if budget <= 0:
        raise ValueError(
            f"token budget {budget} <= 0 (max_model_len={max_model_len} "
            f"gen_headroom={gen_headroom} margin={margin})"
        )
    return budget


# ---------------------------------------------------------------------------
# Per-source staging (wraps CS._stream_stage; per-filter reject counters +
# fingerprint-gated resume + fail-loud on kept==0 come for free).
# ---------------------------------------------------------------------------


@functools.cache
def _resolve_dataset_revision(dataset_id: str, revision_ref: str | None = None) -> str:
    """Pin the upstream HF dataset revision ONCE per (process, dataset, ref).

    The sha enters BOTH the resume fingerprint (a revision bump between crash
    and resume ⇒ fingerprint mismatch ⇒ restage — never a silent stale reuse
    and never a misaligned ``skip_scanned`` fast-forward) AND the
    ``_hf_stream`` call (a resumed stream reads the SAME bytes the checkpoint
    scanned). Fail loud — no fallback to un-pinned 'main' (g1 M2(a)).
    ``revision_ref`` pins a NON-default branch (round 9:
    ``refs/convert/parquet`` for legacy script datasets).
    """
    from huggingface_hub import HfApi

    info = hub.retry_transient(
        lambda: HfApi().dataset_info(dataset_id, revision=revision_ref),
        what=f"dataset_info:{dataset_id}@{revision_ref or 'default'}",
    )
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha:
        raise RuntimeError(f"could not resolve dataset revision for {dataset_id!r}")
    return sha


def _stage_fingerprint(
    spec: SourceSpec,
    *,
    dataset_id: str,
    config: str | None,
    split: str,
    revision: str,
    fallback: bool,
    keep_cap: int,
    token_filter: TokenLengthFilter | None,
    filter_language: bool,
    stream_cap: int | None,
    data_file: str | None = None,
) -> str:
    """ONE fingerprint builder for primary AND fallback staging.

    Carries every output-affecting key (g1 M2 + codex resume-regime-unbound):
    dataset id + pinned REVISION, config/split/data_dir, filters version,
    token budget + the tokenizer-id SET (the dual-tokenizer filter changes
    which rows survive), language filter, KEEP_CAP (a probe-time cap raise
    must restage, never no-op via complete-pool resume), stream_cap, and the
    fallback flag. The fallback path uses the IDENTICAL key set by
    construction (M2(c) fixed a fallback fingerprint that dropped the recipe
    keys).
    """
    # data_files enters the fingerprint ONLY when set: a conditional key keeps
    # every pre-round-9 train-source fingerprint BYTE-IDENTICAL (same key set,
    # same 'train' value), so a reused out-dir RESUMES the crashed run's
    # already-complete pools (wildchat/lmsys/hh/itw, ~45k rows of streaming)
    # and restages exactly the sources whose (config, split, data_files) spec
    # round 9 changed. Round 10: the PER-ATTEMPT template (unsubstituted —
    # the pinned revision is already its own key) replaces the spec-level
    # read; single-template sources (PIPPA) keep a byte-identical value.
    extra = {} if data_file is None else {"data_files": data_file}
    return CS._fingerprint(
        ds=dataset_id,
        revision=revision,
        config=config,
        split=split,
        data_dir=spec.data_dir,
        filters=KEEP_FILTERS_VERSION,
        token_budget=(token_filter.budget_tokens if token_filter else None),
        tokenizers=(list(token_filter.tokenizer_ids) if token_filter else None),
        language=("english" if (filter_language and spec.filter_language) else None),
        keep_cap=keep_cap,
        stream_cap=stream_cap,
        fallback=fallback,
        **extra,
    )


def _seed_seen_from_partial(out_path: Path, fingerprint: str, seen: set[str]) -> int:
    """Re-seed the within-source ``seen`` dedup set from a matching partial
    checkpoint (g1 m1): ``_stream_stage``'s partial resume fast-forwards
    ``skip_scanned`` rows WITHOUT re-running keep_fn, so without this seed a
    post-resume duplicate of a pre-crash kept row is re-admitted and consumes
    a keep-cap slot. Returns the number of seeded keys."""
    partial = out_path.with_name(out_path.name + ".partial.jsonl")
    pmeta = out_path.with_name(out_path.name + ".partial.meta.json")
    n = 0
    if partial.exists() and pmeta.exists():
        meta = json.loads(pmeta.read_text())
        if meta.get("fingerprint") == fingerprint:
            for r in CS.read_jsonl(partial):
                seen.add(CS.norm_text(r["text"]))
                n += 1
    return n


def _access_error_types() -> tuple[type[BaseException], ...]:
    """Exception classes that mean the PRIMARY dataset could not be ACCESSED
    (gated 403 / missing repo / HF transport) — the ONLY classes the
    gated-read fallback may catch (g1 m6: a broad ``except Exception`` was
    silently rerouting the kept==0 fail-loud and genuine code bugs)."""
    from datasets.exceptions import DataFilesNotFoundError, DatasetNotFoundError
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    # GatedRepoError/RepositoryNotFoundError subclass HfHubHTTPError; listed
    # explicitly for greppability.
    return (
        GatedRepoError,
        RepositoryNotFoundError,
        HfHubHTTPError,
        DatasetNotFoundError,
        DataFilesNotFoundError,
    )


def _is_permanent_access_error(exc: BaseException) -> bool:
    """True ONLY for DEFINITIVE access-denial / not-found classes — the ones
    the gated-source skip/fallback may act on (round-6 concern
    ``transient-http-misclassified-as-access-skip``).

    Transient/server-side failures return False and the caller RE-RAISES
    them (never skip-and-redistribute, never fall over to a fallback dataset
    on infra noise): 408 request-timeout, 429 rate-limit, any 5xx, and every
    status-less / unclassified shape. The bounded transient retry lives at
    the revision-resolution seam (``hub.retry_transient`` around
    ``dataset_info`` in ``_resolve_dataset_revision`` — Retry-After-aware,
    ``EPM_HF_RETRY_BUDGET_S``-walled); an error that still escapes it has
    exhausted its retry budget and must HALT (the crash IS the signal;
    staging is fingerprint-checkpointed, so a resume is cheap).
    """
    from datasets.exceptions import DataFilesNotFoundError, DatasetNotFoundError
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    if isinstance(exc, (GatedRepoError, RepositoryNotFoundError)):
        return True  # typed gated / missing-repo classes
    if isinstance(exc, (DatasetNotFoundError, DataFilesNotFoundError)):
        return True  # datasets-lib typed missing/gated dataset or data files
    if isinstance(exc, HfHubHTTPError):
        # Bare HTTP error: permanent ONLY on definitive auth / not-found
        # codes (401 unauthenticated-gated, 403 forbidden, 404 moved/removed).
        # 408/429/5xx and status-less errors are transient — False.
        code = getattr(getattr(exc, "response", None), "status_code", None)
        return isinstance(code, int) and code in (401, 403, 404)
    return False


def verify_declared_splits(
    sources: tuple[SourceSpec, ...],
    *,
    split_names_fn: Callable[..., list[str]] | None = None,
    revision_fn: Callable[[str, str | None], str] | None = None,
) -> dict:
    """Registry-wide (config, split)-existence preflight (round 9).

    ONE metadata pass over every selected spec BEFORE any staging: resolve
    each declared config's OFFERED splits on the pinned revision and check
    every declared split is offered. ALL defects aggregate into ONE
    RuntimeError — the anti-whack-a-mole property (a relaunch surfaces every
    registry defect at once, never one crash per family; round-9 live crash:
    jbb_behaviors inherited the registry-default 'train' against offered
    ['harmful','benign'] and killed the build mid-staging).

    Policy per error class (mirrors the round-5/6 staging semantics):
    - PERMANENT access errors (gated/not-found) -> recorded UNVERIFIABLE and
      left to the runtime gated-skip/fallback — never a preflight crash.
    - TRANSIENT HF errors (408/429/5xx/status-less) -> RE-RAISE (the crash IS
      the signal; staging is fingerprint-checkpointed, resume is cheap).
    - ValueError/RuntimeError from split resolution (bad config name, dead
      dataset script) -> a DEFECT record (same class as a bad split).
    - ``data_files_template`` sources are split-fixed by construction (the
      packaged json builder serves exactly 'train' for a str data_files);
      any other declared split on them is a defect.
    """
    if split_names_fn is None:
        from datasets import get_dataset_split_names as split_names_fn  # type: ignore[no-redef]
    if revision_fn is None:
        revision_fn = _resolve_dataset_revision
    problems: list[str] = []
    unverifiable: list[str] = []
    n_verified = 0
    for spec in sources:
        if spec.data_files_template is not None:
            bad_fmt = [f for f in _data_files_list(spec) if _probe_file_format(f) is None]
            if spec.splits != ("train",):
                problems.append(
                    f"{spec.source_tag} ({spec.dataset_id}): data_files sources serve "
                    f"exactly the 'train' split; declared splits={list(spec.splits)}"
                )
            elif bad_fmt:
                problems.append(
                    f"{spec.source_tag} ({spec.dataset_id}): unstageable data_files "
                    f"format(s): {bad_fmt}"
                )
            else:
                n_verified += 1
            continue
        for config in spec.configs:
            attempt = f"{spec.source_tag}:{config or 'default'} ({spec.dataset_id})"
            kw = {} if spec.data_dir is None else {"data_dir": spec.data_dir}
            try:
                revision = revision_fn(spec.dataset_id, spec.revision_ref)
                offered = list(split_names_fn(spec.dataset_id, config, revision=revision, **kw))
            except _access_error_types() as exc:
                if _is_permanent_access_error(exc):
                    unverifiable.append(
                        f"{attempt}: gated/inaccessible ({type(exc).__name__}) — the "
                        "runtime gated-skip/fallback owns it"
                    )
                    continue
                raise
            except (ValueError, RuntimeError) as exc:
                problems.append(
                    f"{attempt}: config/split resolution failed — "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
                continue
            missing = [s for s in spec.splits if s not in offered]
            if missing:
                problems.append(
                    f"{attempt}: declared split(s) {missing} not offered; offered={offered}"
                )
            else:
                n_verified += 1
    if problems:
        raise RuntimeError(
            "[preflight] SOURCES registry declares (config, split) pairs the datasets do "
            f"not offer — {len(problems)} defect(s), ALL enumerated below (fix the "
            "registry; never silently remap or skip):\n  - " + "\n  - ".join(problems)
        )
    logger.info(
        "[preflight] declared (config, split) verified for %d source-config attempt(s); "
        "%d unverifiable (gated/inaccessible — runtime skip/fallback owns): %s",
        n_verified,
        len(unverifiable),
        "; ".join(unverifiable) or "none",
    )
    return {"n_verified": n_verified, "unverifiable": unverifiable}


# ---------------------------------------------------------------------------
# Round-10 comprehensive per-source tiny-real STREAMING schema gate.
#
# verify_declared_splits() (round 9) checks METADATA (config/split existence);
# this gate checks the ACTUAL FIELD SCHEMAS by streaming a bounded K real rows
# per (source, config, split[, file]) attempt through the FULL production
# extraction + filter + dedup chain (the same _make_keep_fn /
# _make_prefix_keep_fn the build runs), aggregating ALL failures into ONE
# raise — the anti-whack-a-mole property, generalized from splits to schemas.
# Failure classes:
#   (i)  stream/cast errors (the sycophancy_eval `_cast_table` TypeError) —
#        caught directly on the probed slice, AND predicted for multi-file
#        schema-INFERRING sources by probing each file separately and
#        comparing structural type signatures (datasets infers features from
#        the FIRST file and casts every later file to them, so a later file's
#        new struct key / changed type crashes at the file boundary — far
#        past any bounded row cap);
#   (ii) kept==0 field-mapping bugs (text_fields that miss the real schema —
#        the sycophancy_eval kept=0/12,000; the #1092 data-ingestion class);
#   (iii) any other extraction/streaming error.
# Gated/inaccessible sources are UNVERIFIABLE-not-fatal (the runtime
# skip/fallback owns them); a declared fallback dataset is probed when the
# primary is inaccessible (mirroring stage_source's fallover). TRANSIENT HF
# errors re-raise (the crash IS the signal; the gate is cheap to re-run).
# Content hygiene: signatures carry field NAMES + type names only — never
# row text; exception details are truncated type/cast messages.
# ---------------------------------------------------------------------------

SCHEMA_GATE_ROWS = 150  # rows probed per attempt (brief: K ~ 100-200)
SCHEMA_GATE_MAX_FILES = 8  # per-file fan-out cap for multi-file attempts
_SIG_DEPTH_MAX = 3
_NUMERIC_CLASSES = frozenset({"int", "float"})


def _type_class(v: object) -> str:
    """Coarse structural type class of a python value (content-safe)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, bytes):
        return "bytes"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, (list, tuple)):
        return "list"
    return type(v).__name__


def _value_signature(v: object, depth: int = 0) -> dict:
    """Structural type signature of a value: type class + dict KEY names +
    list element signature, depth-capped. NEVER carries content."""
    cls = _type_class(v)
    sig: dict = {"cls": [cls]}
    if depth >= _SIG_DEPTH_MAX:
        return sig
    if cls == "dict":
        sig["keys"] = {str(k): _value_signature(x, depth + 1) for k, x in v.items()}
    elif cls == "list":
        elem = next((x for x in v if x is not None), None)
        if elem is not None:
            sig["elem"] = _value_signature(elem, depth + 1)
    return sig


def _merge_signature(a: dict | None, b: dict | None) -> dict | None:
    """Union-merge two value signatures (across rows of one probe unit)."""
    if a is None:
        return b
    if b is None:
        return a
    out: dict = {"cls": sorted(set(a["cls"]) | set(b["cls"]))}
    if "keys" in a or "keys" in b:
        ka, kb = a.get("keys", {}), b.get("keys", {})
        out["keys"] = {k: _merge_signature(ka.get(k), kb.get(k)) for k in set(ka) | set(kb)}
    if "elem" in a or "elem" in b:
        merged_elem = _merge_signature(a.get("elem"), b.get("elem"))
        if merged_elem is not None:
            out["elem"] = merged_elem
    return out


def _signature_conflicts(anchor: dict | None, later: dict | None, path: str) -> list[str]:
    """Predicted cast conflicts of a LATER file's signature against the FIRST
    (feature-inference anchor) file's.

    Rules (calibrated to datasets' `_cast_table` behavior — the live round-10
    crash): a later file introducing a struct KEY (or top-level field) absent
    from the anchor crashes the cast; a later file's type CLASS outside the
    anchor's crashes (int/float promotion tolerated; null is a wildcard).
    Keys/fields MISSING from a later file are nullable-filled — not flagged.
    """
    if anchor is None or later is None:
        return []
    out: list[str] = []
    a_cls = {c for c in anchor["cls"] if c != "null"}
    l_cls = {c for c in later["cls"] if c != "null"}
    extra = l_cls - a_cls
    if extra and not (extra <= _NUMERIC_CLASSES and a_cls & _NUMERIC_CLASSES):
        out.append(f"{path}: type {sorted(l_cls)} vs first-file {sorted(a_cls)}")
        return out
    if "keys" in later:
        a_keys = anchor.get("keys", {})
        new_keys = sorted(set(later["keys"]) - set(a_keys))
        if new_keys:
            out.append(f"{path}: struct key(s) {new_keys} absent from the first file")
        for k in sorted(set(later["keys"]) & set(a_keys)):
            out.extend(_signature_conflicts(a_keys[k], later["keys"][k], f"{path}.{k}"))
    if "elem" in later and "elem" in anchor:
        out.extend(_signature_conflicts(anchor["elem"], later["elem"], f"{path}[]"))
    return out


def _signature_mixed(sig: dict | None, path: str) -> list[str]:
    """Fields carrying >=2 non-null type classes WITHIN one probe unit —
    feature inference over them is already unstable."""
    if sig is None:
        return []
    out: list[str] = []
    non_null = [c for c in sig["cls"] if c != "null"]
    if len(non_null) > 1 and set(non_null) != _NUMERIC_CLASSES:
        out.append(f"{path}: mixed types {non_null} within one file")
    for k, ksig in sig.get("keys", {}).items():
        out.extend(_signature_mixed(ksig, f"{path}.{k}"))
    if "elem" in sig:
        out.extend(_signature_mixed(sig["elem"], f"{path}[]"))
    return out


def _top_level_fields(sig: dict | None) -> dict[str, str]:
    """Content-safe field -> type-class map for kept_zero diagnostics."""
    if not sig:
        return {}
    return {k: "/".join(ksig["cls"]) for k, ksig in sig.get("keys", {}).items()}


def _enumerate_split_files(
    dataset_id: str,
    config: str | None,
    split: str,
    revision: str,
    data_dir: str | None,
) -> list[str] | None:
    """Best-effort resolved data-file URLs of one (config, split) attempt via
    the dataset builder (metadata only — no row download). None = could not
    enumerate (script dataset, unusual layout); the caller then probes the
    production combined stream instead, so enumeration failure only reduces
    the gate's cross-file PREDICTION, never its keep-chain coverage."""
    try:
        from datasets import load_dataset_builder

        kw = {} if data_dir is None else {"data_dir": data_dir}
        if config is not None:
            builder = load_dataset_builder(dataset_id, config, revision=revision, **kw)
        else:
            builder = load_dataset_builder(dataset_id, revision=revision, **kw)
        data_files = getattr(builder.config, "data_files", None)
        if not data_files or split not in data_files:
            return None
        return [str(f) for f in data_files[split]]
    except _access_error_types():
        raise  # access semantics belong to the caller (gated skip / fallback)
    except Exception as exc:  # noqa: BLE001 — best-effort metadata enumeration
        logger.info(
            "[schema-gate] %s:%s@%s: data-file enumeration unavailable (%s) — "
            "probing the combined stream",
            dataset_id,
            config or "default",
            split,
            type(exc).__name__,
        )
        return None


def _probe_rows(
    it_factory: Callable[[], object],
    keep_fn: Callable[[dict], tuple[dict | None, str | None]],
    *,
    row_cap: int,
    counters: dict,
) -> dict | None:
    """Stream up to ``row_cap`` rows through the production keep chain,
    merging a structural signature. Returns the merged signature."""
    sig: dict | None = None
    stream = it_factory()
    it = iter(stream)
    n_unit = 0  # per-unit row count (counters are shared ACROSS units)
    try:
        for raw in it:
            n_unit += 1
            counters["scanned"] = counters.get("scanned", 0) + 1
            sig = _merge_signature(sig, _value_signature(raw))
            row, reject = keep_fn(raw)
            if reject is not None:
                counters[reject] = counters.get(reject, 0) + 1
            elif row is not None:
                counters["kept"] = counters.get("kept", 0) + 1
            if n_unit >= row_cap:
                break
    finally:
        # Release the streaming pipeline deterministically (gotchas: datasets
        # streaming iterators surviving to interpreter shutdown SIGABRT).
        close = getattr(it, "close", None)
        if callable(close):
            close()
        del it, stream
    return sig


def verify_source_schemas(
    sources: tuple[SourceSpec, ...],
    *,
    token_filter: TokenLengthFilter | None,
    filter_language: bool,
    rows_per_attempt: int = SCHEMA_GATE_ROWS,
    max_files_per_attempt: int = SCHEMA_GATE_MAX_FILES,
    revision_fn: Callable[[str, str | None], str] | None = None,
    enumerate_files_fn: Callable[..., list[str] | None] | None = None,
    stream_open_fn: Callable[..., object] | None = None,
) -> dict:
    """Registry-wide tiny-real streaming schema gate (round 10) — see the
    section comment above for the contract. Raises ONE aggregated
    RuntimeError listing every failing (source, attempt) with its failure
    kind; returns the per-attempt report dict on PASS."""
    if revision_fn is None:
        revision_fn = _resolve_dataset_revision
    if enumerate_files_fn is None:
        enumerate_files_fn = _enumerate_split_files
    if stream_open_fn is None:
        stream_open_fn = _open_source_stream

    failures: list[str] = []
    unverifiable: list[str] = []
    attempts_report: dict[str, dict] = {}

    def _gate_source(spec: SourceSpec, dataset_id: str, fallback: bool) -> None:
        revision = revision_fn(dataset_id, spec.revision_ref)
        # ONE within-source dedup set SHARED across the source's attempts —
        # staging parity (stage_source's seen_committed accumulates across
        # attempts), so an upstream file that fully duplicates an earlier one
        # reads kept_zero HERE instead of tripping the staging fail-loud at
        # production scale (live catch: model-written-evals shipped
        # philpapers2020 byte-identical to nlp_survey).
        source_seen: set[str] = set()
        for config, split, data_file in _iter_attempts(spec):
            key = f"{spec.source_tag}:{_attempt_key(config, split, data_file)}"
            label = f"{key} ({dataset_id})" + (" [fallback]" if fallback else "")
            if data_file is not None:
                units: list[str | None] = [data_file]
                n_files_total = 1
            else:
                files = enumerate_files_fn(dataset_id, config, split, revision, spec.data_dir)
                inferring = bool(files) and all(_probe_file_format(f) for f in files)
                n_files_total = len(files) if files else 0
                if files and len(files) > 1 and inferring:
                    units = list(files[:max_files_per_attempt])
                else:
                    units = [None]  # production combined stream
            per_unit_cap = max(20, rows_per_attempt // max(1, len(units)))
            counters: dict = {}
            unit_sigs: list[tuple[str, dict | None]] = []
            if spec.cross_query_bank is not None:
                keep_fn = _make_prefix_keep_fn(spec, source_seen)
            else:
                keep_fn = _make_keep_fn(
                    spec, config, dataset_id, token_filter, filter_language, source_seen
                )
            for unit in units:
                unit_name = _file_stem(unit) if unit is not None else "<combined>"
                try:
                    sig = _probe_rows(
                        lambda u=unit: stream_open_fn(
                            spec,
                            config=config,
                            split=split,
                            dataset_id=dataset_id,
                            revision=revision,
                            data_file=u,
                        ),
                        keep_fn,
                        row_cap=per_unit_cap,
                        counters=counters,
                    )
                    unit_sigs.append((unit_name, sig))
                except _access_error_types():
                    raise  # source-level access semantics (gated skip / fallback)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:  # noqa: BLE001 — aggregated fail-loud
                    failures.append(
                        f"{label} unit={unit_name}: stream_error — "
                        f"{type(exc).__name__}: {str(exc)[:220]}"
                    )
            kept = counters.get("kept", 0)
            scanned = counters.get("scanned", 0)
            rejects = {k: v for k, v in counters.items() if k not in ("kept", "scanned")}
            if unit_sigs:
                anchor_name, anchor_sig = unit_sigs[0]
                for mixed in _signature_mixed(anchor_sig, "<row>"):
                    failures.append(f"{label} unit={anchor_name}: type_conflict — {mixed}")
                for unit_name, sig in unit_sigs[1:]:
                    for conflict in _signature_conflicts(anchor_sig, sig, "<row>"):
                        failures.append(
                            f"{label}: cross_file_schema_conflict — file "
                            f"{unit_name!r} vs {anchor_name!r}: {conflict} "
                            "(datasets infers features from the first file and "
                            "casts later files to them — this WILL crash the "
                            "combined stream at the file boundary)"
                        )
                if scanned == 0:
                    failures.append(f"{label}: empty_stream — 0 rows streamed")
                elif kept == 0:
                    failures.append(
                        f"{label}: kept_zero — scanned {scanned}, kept 0 "
                        f"(rejects={rejects}; observed top-level fields="
                        f"{_top_level_fields(anchor_sig)}) — text/conv field "
                        "mapping misses the real schema (#1092 class)"
                    )
            attempts_report[key] = {
                "kept": kept,
                "scanned": scanned,
                "rejects": rejects,
                "units_probed": len(unit_sigs),
                "files_resolved": n_files_total,
                "fallback": fallback,
            }
            logger.info(
                "[schema-gate] %s: kept=%d scanned=%d units=%d files=%d",
                label,
                kept,
                scanned,
                len(unit_sigs),
                n_files_total,
            )

    for spec in sources:
        try:
            _gate_source(spec, spec.dataset_id, fallback=False)
        except _access_error_types() as exc:
            if not _is_permanent_access_error(exc):
                raise  # transient HF noise: the crash IS the signal
            if spec.fallback_dataset_id is not None:
                try:
                    _gate_source(spec, spec.fallback_dataset_id, fallback=True)
                    continue
                except _access_error_types() as fexc:
                    if not _is_permanent_access_error(fexc):
                        raise
                    exc = fexc
            unverifiable.append(
                f"{spec.source_tag} ({spec.dataset_id}): gated/inaccessible "
                f"({type(exc).__name__}) — the runtime gated-skip/fallback owns it"
            )
    if failures:
        raise RuntimeError(
            "[schema-gate] per-source field-schema verification FAILED — "
            f"{len(failures)} defect(s), ALL enumerated below (fix the SOURCES "
            "registry / extraction; never silently skip or under-populate):\n  - "
            + "\n  - ".join(failures)
        )
    logger.info(
        "[schema-gate] verified %d attempt(s) (kept>0 through the production keep "
        "chain, no cast conflicts); %d unverifiable (gated/inaccessible): %s",
        len(attempts_report),
        len(unverifiable),
        "; ".join(unverifiable) or "none",
    )
    return {
        "n_verified": len(attempts_report),
        "attempts": attempts_report,
        "unverifiable": unverifiable,
    }


def _make_keep_fn(
    spec: SourceSpec,
    config: str | None,
    dataset_id: str,
    token_filter: TokenLengthFilter | None,
    filter_language: bool,
    seen: set[str],
) -> Callable[[dict], tuple[dict | None, str | None]]:
    def keep(raw: dict) -> tuple[dict | None, str | None]:
        if filter_language and spec.filter_language:
            lang = _row_language(raw)
            if lang is not None and lang.strip().lower() != "english":
                return None, "non_english"
        text = extract_context(raw, spec)
        if text is None:
            return None, "no_text_field"
        reject = CS.usable_text(text)
        if reject:
            return None, reject
        if token_filter is not None and token_filter.too_long(text):
            return None, "too_long_tokens"
        key = CS.norm_text(text)
        if key in seen:
            return None, "dup_text_within_source"
        seen.add(key)
        if spec.moderation_split:
            flagged = _is_moderation_flagged(raw)
            source_tag = f"{spec.source_tag}__{'flagged' if flagged else 'ordinary'}"
            regime = REGIME_NEAR if flagged else spec.regime_class
        else:
            source_tag = spec.source_tag
            regime = spec.regime_class
        return (
            {
                "text": text,
                "source_tag": source_tag,
                "dataset_id": dataset_id,
                "config": config,
                "regime_class": regime,
                "realism_tier": spec.realism_tier,
            },
            None,
        )

    return keep


def _open_source_stream(
    spec: SourceSpec,
    *,
    config: str | None,
    split: str,
    dataset_id: str,
    revision: str,
    data_file: str | None = None,
) -> object:
    """ONE stream-construction seam shared by staging AND the schema gate.

    ``data_file`` set -> that ONE raw repo file via its packaged builder
    (json/csv), features inferred from THAT file alone (revision pinned
    inside the hf:// path; packaged builders take no ``revision=`` kwarg).
    Otherwise the ordinary dataset-id stream. Both route through
    ``CS._hf_stream`` so smoke/test network fakes stay binding.
    """
    if data_file is not None:
        fmt = _probe_file_format(data_file)
        if fmt is None:
            raise ValueError(f"{spec.source_tag}: unstageable data_files format: {data_file}")
        files = data_file.format(revision=revision) if "{revision}" in data_file else data_file
        return CS._hf_stream(fmt, None, split, data_files=files)
    kw = {} if spec.data_dir is None else {"data_dir": spec.data_dir}
    return CS._hf_stream(dataset_id, config, split, revision=revision, **kw)


def _stage_one_config(
    spec: SourceSpec,
    out_dir: Path,
    token_filter: TokenLengthFilter | None,
    *,
    config: str | None,
    split: str,
    dataset_id: str,
    fallback: bool,
    keep_cap: int,
    stream_cap: int | None,
    filter_language: bool,
    seen_committed: set[str],
    data_file: str | None = None,
) -> tuple[list[dict], dict]:
    """Stage ONE (source, config, split[, data_file], dataset) attempt through
    ``CS._stream_stage``.

    Resolves + pins the dataset revision (threaded into BOTH the fingerprint
    and the stream), builds the shared fingerprint, and seeds the
    within-source dedup set from COMPLETED attempts' pools plus any matching
    partial checkpoint — never from a failed prior attempt's in-memory state
    (each attempt copies ``seen_committed``; g1 m1 fallback sibling).
    """
    revision = _resolve_dataset_revision(dataset_id, spec.revision_ref)
    fp = _stage_fingerprint(
        spec,
        dataset_id=dataset_id,
        config=config,
        split=split,
        revision=revision,
        fallback=fallback,
        keep_cap=keep_cap,
        token_filter=token_filter,
        filter_language=filter_language,
        stream_cap=stream_cap,
        data_file=data_file,
    )
    suffix = "__fb" if fallback else ""
    if data_file is not None:
        out_path = out_dir / "staged" / f"{spec.source_tag}__{_file_stem(data_file)}{suffix}.jsonl"
    else:
        split_part = "" if split == "train" else f"__{split}"
        out_path = (
            out_dir
            / "staged"
            / f"{spec.source_tag}__{config or 'default'}{split_part}{suffix}.jsonl"
        )
    seen = set(seen_committed)
    n_seeded = _seed_seen_from_partial(out_path, fp, seen)
    if n_seeded:
        logger.info(
            "[stage] %s:%s: seeded %d dedup keys from partial checkpoint",
            spec.source_tag,
            _attempt_key(config, split, data_file),
            n_seeded,
        )
    keep = _make_keep_fn(spec, config, dataset_id, token_filter, filter_language, seen)

    def row_iter(cfg: str | None = config, dsid: str = dataset_id, rev: str = revision):
        return _open_source_stream(
            spec, config=cfg, split=split, dataset_id=dsid, revision=rev, data_file=data_file
        )

    label = f"{spec.source_tag}:{_attempt_key(config, split, data_file)}" + (
        ":fallback" if fallback else ""
    )
    try:
        return CS._stream_stage(
            out_path=out_path,
            fingerprint=fp,
            row_iter_factory=row_iter,
            keep_fn=keep,
            keep_cap=keep_cap,
            stream_cap=stream_cap,
            log_label=label,
        )
    except ValueError as exc:
        # Round-9 split-absent LOUD handler (the round-5 gated-skip analogue,
        # for splits): `datasets` raises a bare "Bad split:" / "Unknown
        # split" ValueError naming only the split — re-raise NAMING the
        # source + dataset + config so the SOURCES registry is corrected.
        # NEVER silently skip or remap (a skip changes corpus composition;
        # the crash IS the signal). verify_declared_splits() catches this
        # class registry-wide BEFORE staging; this is defense in depth.
        msg = str(exc)
        if "Bad split" in msg or "Unknown split" in msg:
            raise RuntimeError(
                f"[stage] {label}: declared split {split!r} is not offered by "
                f"{dataset_id!r} (config={config or 'default'}) — a SOURCES "
                f"registry defect; fix the SourceSpec (splits=...), never "
                f"silently skip/remap. Upstream: {msg}"
            ) from exc
        raise


def _attempt_key(config: str | None, split: str, data_file: str | None = None) -> str:
    """Counter/label key for one staging attempt — the file stem for a
    per-file (data_files) attempt (round 10), else the bare config label for
    the common 'train' split (backward-compatible with every pre-round-9
    report consumer), config@split otherwise."""
    if data_file is not None:
        return _file_stem(data_file)
    base = config or "default"
    return base if split == "train" else f"{base}@{split}"


def stage_source(
    spec: SourceSpec,
    out_dir: Path,
    token_filter: TokenLengthFilter | None,
    *,
    stream_cap: int | None,
    filter_language: bool,
    seed: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Stream every (config) of a source, filter+within-source-dedup, return
    (kept_rows, {config: counters}).

    - Family-12 crossed sources (``cross_query_bank``) route to
      ``_stage_crossed_source``.
    - The keep-cap is CUMULATIVE across configs (plan §4 item 3: "keep-cap up
      to N across all configs"; g1 m3): later configs get the REMAINING cap.
    - The gated-read fallback catches ACCESS-class errors only (g1 m6); the
      kept==0 fail-loud and genuine code bugs propagate. Once a config falls
      back, later configs of the same source read the fallback dataset too.
    - A fallback-LESS source whose primary read fails a PERMANENT access
      error (``_is_permanent_access_error``: typed gated/not-found, or bare
      401/403/404) is SKIPPED loud (per-config ``skipped_gated_no_access``
      counter, surfaced in ``build_report``'s ``skipped_sources`` +
      ``source_roster``) — one gated dataset must never crash the whole 150k
      build. TRANSIENT HTTP errors (408/429/5xx/timeout/connection) RE-RAISE
      instead — never a skip, never a fallback fallover (round-6 concern
      transient-http-misclassified-as-access-skip). Non-access errors still
      propagate, and ``run_pipeline`` fails loud when the whole corpus / a
      whole regime-class family collapses to zero staged rows.
    """
    if spec.cross_query_bank is not None:
        return _stage_crossed_source(spec, out_dir, token_filter, stream_cap=stream_cap, seed=seed)
    all_rows: list[dict] = []
    counters: dict[str, dict] = {}
    seen_committed: set[str] = set()  # dedup keys from COMPLETED attempts' kept rows
    use_fallback = False
    # Attempts = configs x splits (round 9: a config's data can live in
    # non-'train' splits — jbb harmful/benign, MASK test, BeaverTails
    # 30k_train), OR one attempt per data FILE for data_files sources
    # (round 10: independent feature inference per file). The keep-cap stays
    # CUMULATIVE across ALL attempts.
    for config, split, data_file in _iter_attempts(spec):
        attempt_key = _attempt_key(config, split, data_file)
        remaining_cap = spec.pre_dedup_cap - len(all_rows)
        if remaining_cap <= 0:
            counters[attempt_key] = {"skipped_keep_cap_exhausted": 1}
            continue
        rows: list[dict] = []
        ctr: dict = {}
        if not use_fallback:
            try:
                rows, ctr = _stage_one_config(
                    spec,
                    out_dir,
                    token_filter,
                    config=config,
                    split=split,
                    dataset_id=spec.dataset_id,
                    fallback=False,
                    keep_cap=remaining_cap,
                    stream_cap=stream_cap,
                    filter_language=filter_language,
                    seen_committed=seen_committed,
                    data_file=data_file,
                )
            except _access_error_types() as exc:
                if not _is_permanent_access_error(exc):
                    # Transient/server-side (408/429/5xx/timeout/connection):
                    # NEVER reclassify as an access denial — no skip, no
                    # fallback fallover, no budget redistribution. The
                    # revision seam already bounded-retried
                    # (hub.retry_transient); fail loud and resume from the
                    # fingerprint checkpoint (round-6 concern
                    # transient-http-misclassified-as-access-skip).
                    raise
                if spec.fallback_dataset_id is None:
                    logger.warning(
                        "[stage] %s:%s: %s gated/inaccessible, no fallback — SKIPPING "
                        "this config (later configs of the source may still stage) (%s)",
                        spec.source_tag,
                        attempt_key,
                        spec.dataset_id,
                        repr(exc)[:300],
                    )
                    counters[attempt_key] = {
                        "skipped_gated_no_access": 1,
                        "dataset_id": spec.dataset_id,
                        "reason": repr(exc)[:300],
                    }
                    continue
                logger.warning(
                    "[stage] %s:%s: primary access failed (%s); falling back to %s",
                    spec.source_tag,
                    attempt_key,
                    repr(exc)[:300],
                    spec.fallback_dataset_id,
                )
                use_fallback = True
        if use_fallback:
            # data_file is None by construction here (_iter_attempts refuses
            # data_files + fallback combinations).
            rows, ctr = _stage_one_config(
                spec,
                out_dir,
                token_filter,
                config=config,
                split=split,
                dataset_id=spec.fallback_dataset_id,
                fallback=True,
                keep_cap=remaining_cap,
                stream_cap=stream_cap,
                filter_language=filter_language,
                seen_committed=seen_committed,
            )
        seen_committed.update(CS.norm_text(r["text"]) for r in rows)
        counters[attempt_key] = ctr
        all_rows.extend(rows)
    return all_rows, counters


# ---------------------------------------------------------------------------
# Family-12 crossing (plan §4 item 12): persona/system-prompt prefixes x the
# fixed in-repo query bank (the #1739 ``build_evil_cross`` pattern).
# ---------------------------------------------------------------------------


def _cross_seed(seed: int, source_tag: str) -> int:
    """Deterministic, machine-stable per-source crossing seed."""
    return int(hashlib.sha256(f"{seed}:{source_tag}".encode()).hexdigest()[:8], 16)


def _make_prefix_keep_fn(
    spec: SourceSpec, seen: set[str]
) -> Callable[[dict], tuple[dict | None, str | None]]:
    """keep_fn for family-12 PREFIX staging: scalar text_fields extraction +
    usable_text + within-source dedup ONLY — the token filter runs later on
    the COMPOSED (prefix + query) text, the actual context."""

    def keep(raw: dict) -> tuple[dict | None, str | None]:
        text = None
        for f in spec.text_fields:
            v = raw.get(f)
            if isinstance(v, str) and v.strip():
                text = v
                break
        if text is None:
            return None, "no_text_field"
        reject = CS.usable_text(text)
        if reject:
            return None, reject
        key = CS.norm_text(text)
        if key in seen:
            return None, "dup_text_within_source"
        seen.add(key)
        return {"text": text}, None

    return keep


def _stage_crossed_source(
    spec: SourceSpec,
    out_dir: Path,
    token_filter: TokenLengthFilter | None,
    *,
    stream_cap: int | None,
    seed: int,
) -> tuple[list[dict], dict[str, dict]]:
    """Family-12 construction (plan §4 item 12): stage persona/system-prompt
    PREFIXES from the source, then CROSS them with the fixed in-repo query
    bank, composing each context as ``<prefix>\\n\\n<query>``.

    Prefix staging rides ``CS._stream_stage`` (revision-pinned fingerprint
    resume + per-filter counters + kept>0 fail-loud). The crossing itself is
    a PURE deterministic function of (staged prefixes, committed bank, seed,
    cap) — recomputed each run, no extra checkpoint needed.

    ACCESS errors here FAIL LOUD by design (no skip/fallback semantics —
    round-5 review standing rec, recorded): none of the crossed prefix
    datasets is gated, the family is the 150k top-up lever (a silent skip
    would trip the budget-reachability gate anyway), and the orchestrator's
    pre-launch real-corpus ``--probe`` exercises this path before GPU spend.
    """
    from explore_persona_space.artifacts import banks

    if spec.cross_query_bank is None or len(spec.configs) != 1 or len(spec.splits) != 1:
        raise ValueError(
            f"{spec.source_tag}: crossed sources take a bank + exactly one config + one split"
        )
    config = spec.configs[0]
    split = spec.splits[0]
    revision = _resolve_dataset_revision(spec.dataset_id, spec.revision_ref)
    fp = CS._fingerprint(
        ds=spec.dataset_id,
        revision=revision,
        config=config,
        split=split,
        data_dir=spec.data_dir,
        filters=PREFIX_FILTERS_VERSION,
        keep_cap=spec.pre_dedup_cap,
        stream_cap=stream_cap,
    )
    out_path = out_dir / "staged" / f"{spec.source_tag}__prefixes.jsonl"
    seen_prefix: set[str] = set()
    _seed_seen_from_partial(out_path, fp, seen_prefix)

    def row_iter(cfg: str | None = config, dsid: str = spec.dataset_id, rev: str = revision):
        kw = {} if spec.data_dir is None else {"data_dir": spec.data_dir}
        return CS._hf_stream(dsid, cfg, split, revision=rev, **kw)

    prefix_rows, prefix_ctr = CS._stream_stage(
        out_path=out_path,
        fingerprint=fp,
        row_iter_factory=row_iter,
        keep_fn=_make_prefix_keep_fn(spec, seen_prefix),
        keep_cap=spec.pre_dedup_cap,
        stream_cap=stream_cap,
        log_label=f"{spec.source_tag}:prefixes",
    )
    queries = banks.load_bank(spec.cross_query_bank)
    rows, cross_ctr = cross_with_bank(
        [r["text"] for r in prefix_rows],
        queries,
        spec,
        cap=spec.pre_dedup_cap,
        seed=_cross_seed(seed, spec.source_tag),
        token_filter=token_filter,
    )
    cross_ctr["bank"] = spec.cross_query_bank
    cross_ctr["bank_sha"] = banks.bank_sha(spec.cross_query_bank)
    return rows, {"prefixes": prefix_ctr, "cross": cross_ctr}


def cross_with_bank(
    prefixes: list[str],
    queries: tuple[str, ...] | list[str],
    spec: SourceSpec,
    *,
    cap: int,
    seed: int,
    token_filter: TokenLengthFilter | None,
) -> tuple[list[dict], dict]:
    """Seeded (prefix x query) cross, composed context = prefix + CROSS_JOIN
    + query (the #1739 ``build_evil_cross`` pattern, adapted to the corpus
    row schema).

    Deterministic given (prefixes, queries, seed, cap): prefixes sort by
    content sha (machine-stable, decoupled from stream order); pairs walk a
    seeded permutation of the full n_p x n_q grid until ``cap`` composed rows
    survive the filters (usable_text + dual-tokenizer budget + within-source
    dedup). Fail loud on 0 kept rows.
    """
    if not prefixes or not queries:
        raise RuntimeError(
            f"{spec.source_tag}: empty prefixes ({len(prefixes)}) or bank ({len(queries)})"
        )
    prefixes = sorted(prefixes, key=_context_sha)
    n_p, n_q = len(prefixes), len(queries)
    n_pairs = n_p * n_q
    # Deferred: the ONLY numpy site — a module-top numpy import would freeze
    # BLAS pools before main()'s load_dotenv() binds the shared-VM thread
    # caps (#847 invariant, tests/test_shared_vm_thread_caps.py).
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_pairs)
    kept: list[dict] = []
    counters: dict = {"scanned": 0, "n_prefixes": n_p, "n_queries": n_q, "n_pairs": n_pairs}
    seen: set[str] = set()
    for flat in order:
        if len(kept) >= cap:
            break
        counters["scanned"] += 1
        pi, qi = divmod(int(flat), n_q)
        text = f"{prefixes[pi]}{CROSS_JOIN}{queries[qi]}"
        reject = CS.usable_text(text)
        if reject:
            counters[reject] = counters.get(reject, 0) + 1
            continue
        if token_filter is not None and token_filter.too_long(text):
            counters["too_long_tokens"] = counters.get("too_long_tokens", 0) + 1
            continue
        key = CS.norm_text(text)
        if key in seen:
            counters["dup_text_within_source"] = counters.get("dup_text_within_source", 0) + 1
            continue
        seen.add(key)
        kept.append(
            {
                "text": text,
                "source_tag": spec.source_tag,
                "dataset_id": spec.dataset_id,
                "config": f"{spec.configs[0] or 'default'} x bank:{spec.cross_query_bank}",
                "regime_class": spec.regime_class,
                "realism_tier": spec.realism_tier,
            }
        )
    if not kept:
        rejects = {
            k: v
            for k, v in counters.items()
            if k not in ("scanned", "n_prefixes", "n_queries", "n_pairs")
        }
        raise RuntimeError(
            f"[cross] {spec.source_tag}: kept 0 composed rows after scanning "
            f"{counters['scanned']} pairs (rejects={rejects}) — fail loud"
        )
    counters["n_kept"] = len(kept)
    logger.info(
        "[cross] %s: %d prefixes x %d bank queries -> kept %d composed contexts (scanned %d)",
        spec.source_tag,
        n_p,
        n_q,
        len(kept),
        counters["scanned"],
    )
    return kept, counters


# ---------------------------------------------------------------------------
# Two-stage dedup (MF-K): LSH candidate (16x4 banding) -> exact char-5-gram
# Jaccard >= 0.8 confirm, WITHIN and ACROSS all sources, on CONTEXT text.
# Reproduces FC.exact_jaccard semantics (issue1775_fold_check.py:159) without
# the heavy torch/vLLM import chain that module carries.
# ---------------------------------------------------------------------------


def _char_ngrams(norm: str, n: int = NEAR_DUPE_NGRAM) -> frozenset[str]:
    """Char-n-gram SET of a normalized text (len<n -> {norm} or empty; matches
    issue779_ffc_n1m_generate_capture._char_ngrams / FC semantics)."""
    if len(norm) < n:
        return frozenset([norm]) if norm else frozenset()
    return frozenset(norm[i : i + n] for i in range(len(norm) - n + 1))


def _exact_jaccard(ga: frozenset[str], gb: frozenset[str]) -> float:
    """Exact char-n-gram Jaccard of two n-gram sets (matches FC.exact_jaccard)."""
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga) + len(gb) - inter
    return inter / union if union else 0.0


def dedup_contexts(rows: list[dict]) -> tuple[list[dict], dict]:
    """Incremental within+across-source dedup: keep the FIRST occurrence, drop
    any later context confirmed a near-dup (exact-Jaccard >= 0.8) of an earlier
    kept one whose LSH bands collide. Returns (kept_rows, report_counts).

    Memory: norms + MinHash sigs + a per-band bucket index; n-gram sets are
    computed on demand (LSH candidates are sparse), so peak RSS stays O(pool)
    not O(pool * ngrams).
    """
    n = len(rows)
    if n == 0:
        return [], {"n_in": 0, "n_kept": 0, "n_had_lsh_candidate": 0, "n_confirmed_dropped": 0}
    texts = [r["text"] for r in rows]
    norms = [CS.norm_text(t) for t in texts]
    sigs = CS.minhash_signatures(texts, n_perm=CS.MINHASH_N_PERM, seed=0)  # (n, 64)
    bands = CS.MINHASH_BANDS
    rpb = CS.MINHASH_N_PERM // bands  # 4
    band_index: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]
    kept: list[dict] = []
    n_had_candidate = 0
    n_confirmed = 0
    per_source_dropped: dict[str, int] = defaultdict(int)
    for i in range(n):
        keys = [sigs[i, b * rpb : (b + 1) * rpb].tobytes() for b in range(bands)]
        cand: set[int] = set()
        for b in range(bands):
            cand.update(band_index[b].get(keys[b], ()))
        is_dup = False
        if cand:
            n_had_candidate += 1
            gi = _char_ngrams(norms[i])
            for j in cand:
                if _exact_jaccard(gi, _char_ngrams(norms[j])) >= NEAR_DUPE_JACCARD:
                    is_dup = True
                    break
        if is_dup:
            n_confirmed += 1
            per_source_dropped[rows[i]["source_tag"]] += 1
        else:
            kept.append(rows[i])
            # Register this kept row's band keys under its ORIGINAL index i, so a
            # later candidate confirms exact-Jaccard against norms[i].
            for b in range(bands):
                band_index[b].setdefault(keys[b], []).append(i)
    report = {
        "n_in": n,
        "n_kept": len(kept),
        "n_had_lsh_candidate": n_had_candidate,
        "n_confirmed_dropped": n_confirmed,
        "n_lsh_false_positive": n_had_candidate - n_confirmed,
        "per_source_dropped": dict(per_source_dropped),
        "criterion": {
            "candidate": f"MinHash 16x4 banding over {CS.MINHASH_N_PERM} perms (est>={MINHASH_CAND_EST})",
            "confirm": f"exact char-{NEAR_DUPE_NGRAM}-gram Jaccard >= {NEAR_DUPE_JACCARD}",
            "scope": "WITHIN and ACROSS all sources, on context text, BEFORE split",
        },
    }
    return kept, report


# ---------------------------------------------------------------------------
# Budget re-scaling (plan §4): re-scale the 150k budget proportionally against
# realized per-source post-dedup yields (water-filling, caps preserved).
# ---------------------------------------------------------------------------


def allocate_budget(yields: dict[str, int], caps: dict[str, int], budget: int) -> dict[str, int]:
    """Allocate `budget` across sources proportional to pre-dedup caps, capped
    by realized post-dedup yields; water-fill slack from yield-limited sources
    to those with headroom.

    EXACT: sum(alloc) == min(budget, sum(yields)) — asserted. Grants FLOOR the
    proportional want (never round: independent `round()` grants overshot the
    budget by up to ~n_active/2 rows, g1 m2); the sub-1-row tail is handed out
    1-by-1 in deterministic weight-descending order.
    """
    total = sum(yields.values())
    if total <= budget:
        return dict(yields)
    alloc: dict[str, int] = {t: 0 for t in yields}
    remaining = budget
    active = {t for t in yields if yields[t] > 0}
    while remaining > 0 and active:
        wsum = sum(caps.get(t, yields[t]) for t in active)
        if wsum <= 0:
            break
        grants: dict[str, int] = {}
        for t in active:
            want = remaining * caps.get(t, yields[t]) / wsum
            # FLOOR, never round: sum(floors) <= remaining by construction.
            grants[t] = min(int(want), yields[t] - alloc[t])
        granted = sum(g for g in grants.values() if g > 0)
        if granted <= 0:
            # Every floor is 0 => remaining < n_active: hand out 1 each in
            # deterministic weight-descending (then name) order until spent.
            for t in sorted(active, key=lambda x: (-caps.get(x, yields[x]), x)):
                if remaining <= 0:
                    break
                if alloc[t] < yields[t]:
                    alloc[t] += 1
                    remaining -= 1
            break
        for t, g in grants.items():
            if g > 0:
                alloc[t] += g
        remaining = budget - sum(alloc.values())
        active = {t for t in active if alloc[t] < yields[t]}
    got, want_total = sum(alloc.values()), min(budget, total)
    assert got == want_total, f"allocate_budget inexact: {got} != {want_total}"
    return alloc


def allocate_with_topup(
    yields: dict[str, int],
    caps: dict[str, int],
    topup_tags: set[str],
    budget: int,
) -> dict[str, int]:
    """Plan §4 item 12 top-up lever: non-topup sources allocate FIRST (each
    takes its realized yield when the non-topup total fits the budget, else
    the cap-proportional trim); topup DOMAIN FILLERS then absorb exactly the
    REMAINING budget to 150k, cap-proportionally, bounded by their realized
    yields. Restores the "domain fillers — remaining budget to 150k" lever
    the fixed per-source caps had removed (g1 M1)."""
    base_yields = {t: y for t, y in yields.items() if t not in topup_tags}
    top_yields = {t: y for t, y in yields.items() if t in topup_tags}
    base_alloc = allocate_budget(base_yields, caps, budget)
    remaining = budget - sum(base_alloc.values())
    top_alloc = allocate_budget(top_yields, caps, max(0, remaining))
    return {**base_alloc, **top_alloc}


# ---------------------------------------------------------------------------
# Split (per-source stratified 70/15/15) + LODO group (== source_tag).
# ---------------------------------------------------------------------------


def _context_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_splits(rows: list[dict]) -> None:
    """In-place: per source_tag, deterministically assign train/val/test by
    sha256(text) ordering (exact per-source proportions, order-independent).
    Also sets `context_id`, `context_sha`, and `lodo_group` (== source_tag)."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r["context_sha"] = _context_sha(r["text"])
        r["context_id"] = r["context_sha"][:16]
        r["lodo_group"] = r["source_tag"]
        by_source[r["source_tag"]].append(r)
    for _tag, srows in by_source.items():
        srows.sort(key=lambda r: r["context_sha"])
        n = len(srows)
        n_train = round(SPLIT_FRACS["train"] * n)
        n_val = round(SPLIT_FRACS["val"] * n)
        for idx, r in enumerate(srows):
            if idx < n_train:
                r["split"] = "train"
            elif idx < n_train + n_val:
                r["split"] = "val"
            else:
                r["split"] = "test"


def assert_split_disjoint(rows: list[dict]) -> None:
    """Safety net: no test/val context normalized-text-matches a train context
    (the global dedup already removes cross-split near-dups; this asserts the
    exact-match residue is clean)."""
    train_norm = {CS.norm_text(r["text"]) for r in rows if r["split"] == "train"}
    for r in rows:
        if r["split"] in ("val", "test") and CS.norm_text(r["text"]) in train_norm:
            raise RuntimeError(
                f"split leakage: a {r['split']} context (source {r['source_tag']}) "
                "exact-matches a train context after global dedup — fail loud"
            )


# ---------------------------------------------------------------------------
# Reports (counts only — never item text).
# ---------------------------------------------------------------------------


def build_source_roster(
    sources: tuple[SourceSpec, ...],
    *,
    pre_dedup_per_source: dict[str, int],
    post_dedup_per_source: dict[str, int],
    allocation: dict[str, int],
    final_per_source: dict[str, int],
    skipped_sources: list[dict],
) -> list[dict]:
    """Realized per-source composition roster (round-6 concern
    ``gated-sources-absent-corpus-composition``): ONE row per SELECTED spec —
    status (staged / skipped_gated_no_access / partially_skipped), planned
    pre-dedup cap, and per-regime-tag realized counts at every stage
    (pre-dedup / post-dedup / allocated / final) — so a production build with
    absent gated sources DISCLOSES its composition durably instead of
    shifting silently. Counts only — never item text."""
    skipped_by_tag: dict[str, list[str]] = defaultdict(list)
    for s in skipped_sources:
        skipped_by_tag[s["source_tag"]].append(s["config"])
    roster: list[dict] = []
    for spec in sources:
        tags = sorted(_regime_tags(spec))
        pre_total = sum(pre_dedup_per_source.get(t, 0) for t in tags)
        skipped_cfgs = sorted(skipped_by_tag.get(spec.source_tag, []))
        if skipped_cfgs and pre_total == 0:
            status = "skipped_gated_no_access"
        elif skipped_cfgs:
            status = "partially_skipped"
        else:
            status = "staged"
        roster.append(
            {
                "source_tag": spec.source_tag,
                "dataset_id": spec.dataset_id,
                "fallback_dataset_id": spec.fallback_dataset_id,
                "regime_tags": _regime_tags(spec),
                "planned_pre_dedup_cap": spec.pre_dedup_cap,
                "topup": spec.topup,
                "status": status,
                "skipped_configs": skipped_cfgs,
                "pre_dedup_rows": {t: pre_dedup_per_source.get(t, 0) for t in tags},
                "post_dedup_rows": {t: post_dedup_per_source.get(t, 0) for t in tags},
                "allocated": {t: allocation.get(t, 0) for t in tags},
                "final_rows": {t: final_per_source.get(t, 0) for t in tags},
            }
        )
    return roster


def build_report(
    *,
    pre_dedup_per_source: dict[str, int],
    dedup_report: dict,
    post_dedup_per_source: dict[str, int],
    allocation: dict[str, int],
    final_rows: list[dict],
    regime_table: dict[str, str],
    stream_counters: dict[str, dict],
    budget: int,
    skipped_sources: list[dict] | None = None,
    sources: tuple[SourceSpec, ...] | None = None,
) -> dict:
    split_counts: dict[str, int] = defaultdict(int)
    regime_counts: dict[str, int] = defaultdict(int)
    tier_counts: dict[int, int] = defaultdict(int)
    final_per_source: dict[str, int] = defaultdict(int)
    for r in final_rows:
        # Probe mode reports realized post-dedup yields BEFORE assign_splits
        # runs, so rows carry no "split" yet (u4 smoke catch: a bare
        # r["split"] KeyError'd every --probe invocation).
        split_counts[r.get("split", "unassigned")] += 1
        regime_counts[r["regime_class"]] += 1
        tier_counts[r["realism_tier"]] += 1
        final_per_source[r["source_tag"]] += 1
    return {
        "budget": budget,
        "n_final": len(final_rows),
        "pre_dedup_per_source": pre_dedup_per_source,
        "post_dedup_per_source": post_dedup_per_source,
        "final_per_source": dict(final_per_source),
        "budget_allocation": allocation,
        "budget_allocation_total": sum(allocation.values()),
        "budget_shortfall": max(0, budget - sum(allocation.values())),
        "split_counts": dict(split_counts),
        "regime_class_counts": dict(regime_counts),
        "realism_tier_counts": {str(k): v for k, v in tier_counts.items()},
        "dedup": dedup_report,
        "stream_counters": stream_counters,
        "skipped_sources": list(skipped_sources or []),
        # Round-6 composition disclosure (gated-sources-absent-corpus-composition):
        # realized per-source roster + where a skipped source's budget went.
        "source_roster": build_source_roster(
            sources or (),
            pre_dedup_per_source=pre_dedup_per_source,
            post_dedup_per_source=post_dedup_per_source,
            allocation=allocation,
            final_per_source=dict(final_per_source),
            skipped_sources=list(skipped_sources or []),
        ),
        "budget_redistribution": {
            "skipped_source_tags": sorted({s["source_tag"] for s in (skipped_sources or [])}),
            "skipped_planned_caps": {
                spec.source_tag: spec.pre_dedup_cap
                for spec in (sources or ())
                if any(s["source_tag"] == spec.source_tag for s in (skipped_sources or []))
            },
            "topup_tags": sorted(
                tag for spec in (sources or ()) if spec.topup for tag in _regime_tags(spec)
            ),
            "note": (
                "budget re-scales over realized post-dedup yields of SURVIVING "
                "sources only (allocate_with_topup); a skipped source contributes "
                "no yields, so its planned share flows to the survivors' "
                "proportional re-scale, with topup sources absorbing the residual "
                "to the 150k target — the plan-v7-sanctioned behavior (plan §4: "
                "'the P0 --probe re-scales the 150k budget proportionally against "
                "realized per-source counts'); a material shortfall past the "
                "top-up lever still HALTS via the budget-reachability gate"
            ),
        },
        "source_regime_table": regime_table,
        "weight_pct": {
            "weird_ood": round(
                100
                * sum(
                    v
                    for k, v in regime_counts.items()
                    if k in (REGIME_WEIRD, REGIME_NEAR, REGIME_IDIO)
                )
                / max(1, len(final_rows)),
                1,
            ),
            "ordinary": round(
                100 * regime_counts.get(REGIME_ORDINARY, 0) / max(1, len(final_rows)), 1
            ),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def _selected_sources(names: list[str] | None) -> tuple[SourceSpec, ...]:
    if not names:
        return SOURCES
    wanted = set(names)
    sel = tuple(s for s in SOURCES if s.source_tag in wanted)
    missing = wanted - {s.source_tag for s in sel}
    if missing:
        raise SystemExit(f"unknown --sources tags: {sorted(missing)}")
    return sel


def _log_roster(report: dict) -> None:
    """One loud composition line per run (never silent): staged / partial /
    skipped source counts, from the durable ``source_roster`` report key."""
    roster = report.get("source_roster", [])
    by_status: dict[str, int] = defaultdict(int)
    for row in roster:
        by_status[row["status"]] += 1
    logger.info(
        "[corpus] source roster: %d staged / %d partially_skipped / %d "
        "skipped_gated_no_access of %d selected sources (durable in "
        "report['source_roster'] + ['budget_redistribution'])",
        by_status.get("staged", 0),
        by_status.get("partially_skipped", 0),
        by_status.get("skipped_gated_no_access", 0),
        len(roster),
    )


def run_pipeline(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = _selected_sources(args.sources)
    regime_table = build_source_regime_table(sources)

    # Round-9 registry-wide preflight: every declared (config, split) must be
    # offered BEFORE any staging spend — one pass, ALL defects enumerated.
    if args.skip_split_preflight:
        logger.warning("[preflight] --skip-split-preflight: declared (config, split) NOT verified")
    else:
        verify_declared_splits(sources)

    token_filter = None
    if not args.no_token_filter:
        budget_tokens = token_budget(args.max_model_len, args.gen_headroom, args.token_margin)
        token_filter = TokenLengthFilter(tuple(args.tokenizers), budget_tokens)
        logger.info(
            "[corpus] token filter (fit-BOTH-models): %s budget=%d tokens",
            ",".join(token_filter.tokenizer_ids),
            budget_tokens,
        )

    # Round-10 comprehensive schema gate: stream K real rows per attempt
    # through the FULL production keep chain BEFORE any staging spend — one
    # pass, ALL field-schema defects (cast crashes, kept==0 mappings)
    # enumerated at once (the split preflight's schema-level sibling).
    if args.skip_schema_gate:
        if args.schema_gate_only:
            raise SystemExit("--schema-gate-only contradicts --skip-schema-gate")
        logger.warning("[schema-gate] --skip-schema-gate: per-source field schemas NOT verified")
    else:
        gate_report = verify_source_schemas(
            sources,
            token_filter=token_filter,
            filter_language=not args.no_language_filter,
            rows_per_attempt=args.schema_gate_rows,
        )
        if args.schema_gate_only:
            gate_report["mode"] = "schema-gate-only"
            _write_json(out_dir / "schema_gate_report.json", gate_report)
            logger.info(
                "[schema-gate] GATE-ONLY complete -> %s", out_dir / "schema_gate_report.json"
            )
            return gate_report

    # 1. Stage every source (streaming, checkpointed, fail-loud on kept==0 for
    # ACCESSIBLE sources; a gated fallback-less source is SKIPPED loud and
    # recorded — see stage_source + the aggregate guards below).
    all_rows: list[dict] = []
    pre_dedup_per_source: dict[str, int] = defaultdict(int)
    stream_counters: dict[str, dict] = {}
    skipped_sources: list[dict] = []
    for spec in sources:
        rows, ctr = stage_source(
            spec,
            out_dir,
            token_filter,
            stream_cap=args.stream_cap,
            filter_language=not args.no_language_filter,
            seed=args.seed,
        )
        for r in rows:
            pre_dedup_per_source[r["source_tag"]] += 1
        stream_counters[spec.source_tag] = ctr
        for cfg_name, cfg_ctr in ctr.items():
            if cfg_ctr.get("skipped_gated_no_access"):
                skipped_sources.append(
                    {
                        "source_tag": spec.source_tag,
                        "dataset_id": cfg_ctr.get("dataset_id", spec.dataset_id),
                        "config": cfg_name,
                        "reason": cfg_ctr.get("reason", "gated/inaccessible, no fallback"),
                    }
                )
        all_rows.extend(rows)
    logger.info("[corpus] staged %d rows across %d sources", len(all_rows), len(sources))
    if skipped_sources:
        logger.warning(
            "[corpus] %d source config(s) SKIPPED (gated/inaccessible, no fallback): %s",
            len(skipped_sources),
            "; ".join(
                f"{s['source_tag']}:{s['config']} ({s['dataset_id']})" for s in skipped_sources
            ),
        )
    # Aggregate fail-loud guards: a per-source skip is tolerable; a corpus with
    # NOTHING staged, or a whole regime-class family collapsing to zero rows,
    # is a real problem worth halting on (never a silently thin build).
    if not all_rows:
        raise RuntimeError(
            f"corpus staging kept 0 rows across all {len(sources)} selected sources "
            f"(skipped gated/no-fallback: {sorted({s['source_tag'] for s in skipped_sources})}) "
            "— fail loud"
        )
    staged_regimes = {r["regime_class"] for r in all_rows}
    # Declared classes come from the REGIME TABLE (per realized tag), not bare
    # spec.regime_class — a moderation-split source ALSO declares its derived
    # near-distribution stratum, so a selective --sources probe cannot lose it
    # silently (round-6 NIT moderation-derived-regime-guard-gap).
    missing_regimes = sorted(set(regime_table.values()) - staged_regimes)
    if missing_regimes:
        raise RuntimeError(
            f"regime class(es) {missing_regimes} kept 0 staged rows "
            f"(skipped gated/no-fallback: {sorted({s['source_tag'] for s in skipped_sources})}) "
            "— a whole regime family collapsing to nothing is a real problem; fail loud"
        )

    # 2. Two-stage dedup WITHIN+ACROSS sources, BEFORE split (MF-K).
    deduped, dedup_report = dedup_contexts(all_rows)
    post_dedup_per_source: dict[str, int] = defaultdict(int)
    for r in deduped:
        post_dedup_per_source[r["source_tag"]] += 1
    # Fail loud on any source that had rows pre-dedup but 0 post-dedup.
    for tag, pre in pre_dedup_per_source.items():
        if pre > 0 and post_dedup_per_source.get(tag, 0) == 0:
            raise RuntimeError(
                f"source {tag}: {pre} rows pre-dedup, 0 post-dedup — total-overlap "
                "or a dedup bug; fail loud (MF-K)"
            )
    logger.info(
        "[corpus] dedup: %d -> %d (candidate=%d confirmed_dropped=%d)",
        dedup_report["n_in"],
        dedup_report["n_kept"],
        dedup_report["n_had_lsh_candidate"],
        dedup_report["n_confirmed_dropped"],
    )

    # 3. Re-scale budget against realized per-source post-dedup yields:
    # non-topup sources first, then the family-12 domain-filler top-up lever
    # ("remaining budget to 150k", plan §4 item 12). A moderation-split
    # family's weight is SPLIT across its substrata so the family carries its
    # plan budget ONCE, not twice (g1 m3).
    caps: dict[str, int] = {}
    topup_tags: set[str] = set()
    for spec in sources:
        tags = _regime_tags(spec)
        for tag in tags:
            caps[tag] = max(1, spec.pre_dedup_cap // len(tags))
            if spec.topup:
                topup_tags.add(tag)
    allocation = allocate_with_topup(dict(post_dedup_per_source), caps, topup_tags, args.budget)
    total_alloc = sum(allocation.values())
    shortfall = max(0, args.budget - total_alloc)
    # Budget-reachability gate: the FULL production build (all sources, no
    # stream cap) must reach the 150k target — a shortfall past the top-up
    # lever means realized yields collapsed and needs a re-plan, never a
    # silently short corpus. Demoted to a log line on probe / subset /
    # stream-capped (smoke) runs (gate-calibration parity, gotchas.md).
    full_production_shape = args.sources is None and args.stream_cap is None
    if shortfall > 0:
        msg = (
            f"[corpus] budget shortfall: allocation {total_alloc} < budget {args.budget} "
            f"(shortfall {shortfall}) even after the domain-filler top-up lever"
        )
        if full_production_shape and not args.probe:
            raise RuntimeError(msg + " — realized yields too low; surface for re-plan")
        logger.info("%s (non-production shape: reported, not fatal)", msg)

    # 4. Every realized source_tag must resolve to a committed regime class.
    unknown = sorted({r["source_tag"] for r in deduped} - set(regime_table))
    if unknown:
        raise RuntimeError(f"unknown source_tag(s) not in SOURCE_REGIME: {unknown}")

    if args.probe:
        report = build_report(
            pre_dedup_per_source=dict(pre_dedup_per_source),
            dedup_report=dedup_report,
            post_dedup_per_source=dict(post_dedup_per_source),
            allocation=allocation,
            final_rows=deduped,  # probe reports realized yields, not the trim
            regime_table=regime_table,
            stream_counters=stream_counters,
            budget=args.budget,
            skipped_sources=skipped_sources,
            sources=sources,
        )
        _log_roster(report)
        report["mode"] = "probe"
        report["note"] = "post-dedup yields + re-scaled budget allocation (no full write)"
        _write_json(out_dir / "probe_report.json", report)
        logger.info("[corpus] PROBE complete -> %s", out_dir / "probe_report.json")
        return report

    # 5. Subsample each source to its allocation (seeded), then split + LODO.
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in deduped:
        by_source[r["source_tag"]].append(r)
    final_rows: list[dict] = []
    for tag, srows in by_source.items():
        final_rows.extend(CS._subsample(srows, allocation.get(tag, len(srows)), args.seed))
    assign_splits(final_rows)
    assert_split_disjoint(final_rows)

    # 6. Persist the corpus JSONL + committed regime table + dedup report.
    corpus_path = out_dir / "corpus.jsonl"
    CS._write_jsonl_atomic(
        corpus_path,
        [
            {
                "context_id": r["context_id"],
                "context_sha": r["context_sha"],
                "text": r["text"],
                "source_tag": r["source_tag"],
                "dataset_id": r["dataset_id"],
                "config": r["config"],
                "regime_class": r["regime_class"],
                "realism_tier": r["realism_tier"],
                "split": r["split"],
                "lodo_group": r["lodo_group"],
            }
            for r in final_rows
        ],
    )
    _write_json(out_dir / "source_regime_table.json", regime_table)
    report = build_report(
        pre_dedup_per_source=dict(pre_dedup_per_source),
        dedup_report=dedup_report,
        post_dedup_per_source=dict(post_dedup_per_source),
        allocation=allocation,
        final_rows=final_rows,
        regime_table=regime_table,
        stream_counters=stream_counters,
        budget=args.budget,
        skipped_sources=skipped_sources,
        sources=sources,
    )
    _log_roster(report)
    report["mode"] = "build"
    report["corpus_path"] = str(corpus_path)
    _write_json(out_dir / "dedup_report.json", report)
    logger.info("[corpus] BUILD complete: %d rows -> %s", len(final_rows), corpus_path)

    # 7. Optional HF upload (text/JSON, non-LFS path; unconditional-persist).
    if args.upload:
        load_dotenv()
        prefix = args.upload_prefix.rstrip("/")
        for local in (
            corpus_path,
            out_dir / "source_regime_table.json",
            out_dir / "dedup_report.json",
        ):
            dest = f"{prefix}/{local.name}"
            uploaded = hub.upload_dataset(
                data_path=str(local), repo_id=args.upload_repo, path_in_repo=dest
            )
            if not uploaded:
                raise RuntimeError(f"HF upload failed for {local} -> {dest}")
            logger.info("[corpus] uploaded %s", uploaded)
    return report


def _write_json(path: Path, obj: dict) -> None:
    """Atomic JSON write with a PROCESS-UNIQUE temp name in the destination's
    own directory (#2336: a fixed ``<name>.tmp`` is process-shared — two
    concurrent writers of one destination collide mid-``os.replace``); on any
    failure the temp is best-effort unlinked and the ORIGINAL exception
    propagates."""
    import os
    import uuid

    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            logger.warning("[corpus] temp cleanup failed for %s", tmp)
        raise


# ---------------------------------------------------------------------------
# Synthetic selfcheck (no network, no tokenizer download)
# ---------------------------------------------------------------------------


def run_selfcheck() -> None:
    """Synthetic-fixture self-check of the PURE data-plane functions — no
    network, no tokenizer download (fake count fns injected at the
    ``TokenLengthFilter._count_fns`` seam). Raises on any failure."""
    # 1. allocate_budget exactness (g1 m2 probe: independent round() grants
    # overshot budget=5 to 6 on {10,10,10}).
    alloc = allocate_budget({"a": 10, "b": 10, "c": 10}, {"a": 10, "b": 10, "c": 10}, 5)
    assert sum(alloc.values()) == 5, alloc
    for yields, budget in (
        ({"a": 7, "b": 3}, 100),  # under budget -> yields verbatim
        ({"a": 1000, "b": 1, "c": 500}, 900),
        ({f"s{i}": 13 for i in range(35)}, 150),
    ):
        caps = {t: max(1, y) for t, y in yields.items()}
        a = allocate_budget(dict(yields), caps, budget)
        assert sum(a.values()) == min(budget, sum(yields.values())), (a, budget)
        assert all(0 <= a[t] <= yields[t] for t in yields), a

    # 2. topup lever: fillers absorb exactly the remaining budget (plan §4
    # item 12 "remaining budget to 150k"); a base overshoot leaves them 0.
    yields = {"base1": 50, "base2": 30, "fill1": 100, "fill2": 100}
    caps = {"base1": 60, "base2": 40, "fill1": 100, "fill2": 50}
    alloc = allocate_with_topup(yields, caps, {"fill1", "fill2"}, 150)
    assert alloc["base1"] == 50 and alloc["base2"] == 30, alloc
    assert alloc["fill1"] + alloc["fill2"] == 70 and sum(alloc.values()) == 150, alloc
    alloc2 = allocate_with_topup(yields, caps, {"fill1", "fill2"}, 60)
    assert sum(alloc2.values()) == 60 and alloc2["fill1"] == alloc2["fill2"] == 0, alloc2

    # 3. dual-tokenizer JOINT filter: a row passing tokenizer A but exceeding
    # the budget under tokenizer B is REJECTED (fit-BOTH-models semantics).
    filt = TokenLengthFilter(("fake/a", "fake/b"), budget_tokens=10)
    filt._count_fns = [lambda t: len(t.split()), lambda t: 3 * len(t.split())]
    assert not filt.too_long("one two three")  # 3 and 9 both <= 10
    assert filt.too_long("a b c d")  # 4 <= 10 under A but 12 > 10 under B

    # 4. moderation flag: toxic flags; `redacted` alone must NOT (g1 m5).
    assert _is_moderation_flagged({"toxic": True})
    assert not _is_moderation_flagged({"redacted": True})
    assert _is_moderation_flagged({"openai_moderation": [{"flagged": True}]})
    assert not _is_moderation_flagged({"openai_moderation": [{"flagged": False}]})

    # 5. family-12 crossing: composition shape, dedup, cap, determinism,
    # composed-text token filter, kept==0 fail-loud.
    spec = SourceSpec(
        source_tag="selfcheck_prefix",
        dataset_id="selfcheck/prefixes",
        regime_class=REGIME_NEAR,
        realism_tier=3,
        pre_dedup_cap=8,
        cross_query_bank="selfcheck_bank",
    )
    prefixes = [f"prefix persona number {i} with sufficient length" for i in range(6)]
    queries = [f"synthetic query {j} of adequate length?" for j in range(5)]
    rows1, ctr1 = cross_with_bank(list(prefixes), queries, spec, cap=8, seed=7, token_filter=None)
    rows2, _ = cross_with_bank(list(prefixes), queries, spec, cap=8, seed=7, token_filter=None)
    assert [r["text"] for r in rows1] == [r["text"] for r in rows2], "crossing not deterministic"
    assert len(rows1) == 8 and ctr1["n_kept"] == 8, ctr1
    assert all(CROSS_JOIN in r["text"] for r in rows1)
    assert len({r["text"] for r in rows1}) == len(rows1), "crossed rows not unique"
    filt2 = TokenLengthFilter(("fake/a",), budget_tokens=3)
    filt2._count_fns = [lambda t: len(t.split())]
    try:
        cross_with_bank(list(prefixes), queries, spec, cap=8, seed=7, token_filter=filt2)
        raise AssertionError("expected fail-loud on 0 kept crossed rows")
    except RuntimeError as exc:
        assert "kept 0 composed rows" in str(exc)

    # 6. fingerprint regime keys: every output-affecting key flips the fp,
    # and primary/fallback share ONE builder (g1 M2 / resume-regime-unbound).
    base_spec = SOURCES[0]
    filt3 = TokenLengthFilter(tuple(DEFAULT_TOKENIZERS), budget_tokens=100)
    kw: dict = dict(
        dataset_id="d",
        config=None,
        split="train",
        revision="r1",
        fallback=False,
        keep_cap=10,
        token_filter=filt3,
        filter_language=True,
        stream_cap=None,
    )
    fp0 = _stage_fingerprint(base_spec, **kw)
    for delta in (
        {"revision": "r2"},
        {"split": "harmful"},  # round 9: split is a first-class regime key
        {"keep_cap": 11},
        {"fallback": True},
        {"token_filter": TokenLengthFilter(("only/one",), budget_tokens=100)},
        {"stream_cap": 5},
        {"data_file": "hf://datasets/d@{revision}/a.jsonl"},  # round 10: per-file attempts
    ):
        assert _stage_fingerprint(base_spec, **{**kw, **delta}) != fp0, delta

    # 7. split + disjointness on synthetic rows (70/15/15 at n=100).
    srows = [
        {
            "text": f"synthetic split row {i} with plenty of distinct words {i * 7}",
            "source_tag": "s1",
        }
        for i in range(100)
    ]
    assign_splits(srows)
    scounts: dict[str, int] = defaultdict(int)
    for r in srows:
        scounts[r["split"]] += 1
    assert dict(scounts) == {"train": 70, "val": 15, "test": 15}, dict(scounts)
    assert_split_disjoint(srows)

    # 8. dedup: exact dup dropped (candidate + confirm), distinct kept.
    d_rows = [
        {"text": "the same exact duplicated context text for the dedup check", "source_tag": "s1"},
        {"text": "the same exact duplicated context text for the dedup check", "source_tag": "s2"},
        {"text": "a completely different context that must survive dedup here", "source_tag": "s1"},
    ]
    kept, rep = dedup_contexts(d_rows)
    assert rep["n_kept"] == 2 and rep["n_confirmed_dropped"] == 1, rep

    # 9. round-9 split preflight: ALL declared-(config,split) defects
    # aggregate into ONE raise (anti-whack-a-mole); gated sources are
    # unverifiable-not-fatal; injected fns only — no network.
    def _fake_rev(dataset_id: str, revision_ref: str | None = None) -> str:
        from huggingface_hub.utils import GatedRepoError

        if dataset_id == "sc/gated":
            raise GatedRepoError("gated fixture")
        return "0" * 40

    def _fake_split_names(dataset_id, config=None, *, revision=None, **kw):
        if config == "badcfg":
            raise ValueError(f"BuilderConfig 'badcfg' not found. Available: ['default']")
        return ["harmful", "benign"] if dataset_id == "sc/jbb-like" else ["train"]

    def _sc_spec(tag, ds, **kws):
        return SourceSpec(
            source_tag=tag,
            dataset_id=ds,
            regime_class=REGIME_WEIRD,
            realism_tier=2,
            pre_dedup_cap=10,
            text_fields=("text",),
            **kws,
        )

    ok = _sc_spec("sc_ok", "sc/plain")
    bad_split = _sc_spec("sc_badsplit", "sc/jbb-like")  # default 'train' vs harmful/benign
    bad_cfg = _sc_spec("sc_badcfg", "sc/plain", configs=("badcfg",))
    gated = _sc_spec("sc_gated", "sc/gated")
    try:
        verify_declared_splits(
            (ok, bad_split, bad_cfg, gated),
            split_names_fn=_fake_split_names,
            revision_fn=_fake_rev,
        )
        raise AssertionError("expected aggregated preflight raise")
    except RuntimeError as exc:
        msg = str(exc)
        assert "sc_badsplit" in msg and "sc_badcfg" in msg, msg  # BOTH enumerated
        assert "sc_gated" not in msg and "sc_ok" not in msg, msg
    res = verify_declared_splits(
        (ok, gated), split_names_fn=_fake_split_names, revision_fn=_fake_rev
    )
    assert res["n_verified"] == 1 and len(res["unverifiable"]) == 1, res
    fixed = _sc_spec("sc_fixed", "sc/jbb-like", splits=("harmful", "benign"))
    res2 = verify_declared_splits((fixed,), split_names_fn=_fake_split_names, revision_fn=_fake_rev)
    assert res2["n_verified"] == 1, res2

    # 10. round-10 comprehensive schema gate: cross-file struct-key conflict +
    # kept_zero + stream_error aggregate into ONE raise (anti-whack-a-mole,
    # schema level); gated sources unverifiable-not-fatal; the corrected
    # sycophancy-eval turn shape ({"type": "human", "content"}) passes through
    # the REAL production keep chain; per-file attempts for data_files specs.
    from huggingface_hub.utils import GatedRepoError

    syco = next(s for s in SOURCES if s.source_tag == "sycophancy_eval")
    syco_attempts = _iter_attempts(syco)
    assert len(syco_attempts) == 4 and all(a[2] is not None for a in syco_attempts), syco_attempts
    assert _attempt_key(*syco_attempts[0]) == "answer", syco_attempts[0]
    assert first_user_content([{"type": "human", "content": "hello there friend"}]) == (
        "hello there friend"
    )
    assert first_user_content([{"type": "ai", "content": "assistant turn only"}]) is None

    def _sg_rev(dataset_id: str, revision_ref: str | None = None) -> str:
        if dataset_id == "sg/gated":
            raise GatedRepoError("gated fixture")
        return "1" * 40

    def _sg_files(dataset_id, config, split, revision, data_dir):
        return {"sg/multi": ["hf://x/a.jsonl", "hf://x/b.jsonl"]}.get(dataset_id)

    _sg_text = "sufficiently long synthetic schema-gate context text %d"
    sg_rows: dict[tuple[str, str | None], list[dict]] = {
        # the live crash shape: file b's metadata struct adds a key file a lacks
        ("sg/multi", "a"): [
            {"q": _sg_text % i, "metadata": {"prompt_template": "t"}} for i in range(3)
        ],
        ("sg/multi", "b"): [
            {
                "q": _sg_text % (10 + i),
                "metadata": {"prompt_template": "t", "prompt_template_type": "x"},
            }
            for i in range(3)
        ],
        # the live kept=0 shape: scalar text_fields against a LIST-valued field
        ("sg/keptzero", None): [
            {"prompt": [{"type": "human", "content": _sg_text % i}]} for i in range(4)
        ],
        ("sg/conv", None): [
            {"prompt": [{"type": "human", "content": _sg_text % i}]} for i in range(4)
        ],
        ("sg/ok", None): [{"text": _sg_text % i} for i in range(4)],
    }

    def _sg_stream(spec, *, config, split, dataset_id, revision, data_file=None):
        if dataset_id == "sg/crash":
            raise TypeError("Couldn't cast array of type struct<a: string> to Value('string')")
        return iter(sg_rows[(dataset_id, _file_stem(data_file) if data_file else None)])

    def _sg_spec(tag, ds, **kws):
        return SourceSpec(
            source_tag=tag,
            dataset_id=ds,
            regime_class=REGIME_NEAR,
            realism_tier=2,
            pre_dedup_cap=10,
            **kws,
        )

    sg_kwargs = dict(
        token_filter=None,
        filter_language=False,
        revision_fn=_sg_rev,
        enumerate_files_fn=_sg_files,
        stream_open_fn=_sg_stream,
    )
    try:
        verify_source_schemas(
            (
                _sg_spec("sg_multi", "sg/multi", text_fields=("q",)),
                _sg_spec("sg_keptzero", "sg/keptzero", text_fields=("prompt",)),
                _sg_spec("sg_crash", "sg/crash", text_fields=("text",)),
            ),
            **sg_kwargs,
        )
        raise AssertionError("expected aggregated schema-gate raise")
    except RuntimeError as exc:
        msg = str(exc)
        assert "cross_file_schema_conflict" in msg and "prompt_template_type" in msg, msg
        assert "sg_keptzero" in msg and "kept_zero" in msg, msg
        assert "sg_crash" in msg and "stream_error" in msg and "cast" in msg, msg
    res_sg = verify_source_schemas(
        (
            _sg_spec("sg_conv", "sg/conv", conv_fields=("prompt",)),
            _sg_spec("sg_ok", "sg/ok", text_fields=("text",)),
            _sg_spec("sg_gated", "sg/gated", text_fields=("text",)),
        ),
        **sg_kwargs,
    )
    assert res_sg["n_verified"] == 2 and len(res_sg["unverifiable"]) == 1, res_sg
    assert res_sg["attempts"]["sg_conv:default"]["kept"] == 4, res_sg["attempts"]

    print(
        "issue2502_corpus: selfcheck OK (allocation, topup, dual-tokenizer, "
        "moderation-flag, crossing, fingerprint-keys, split, dedup, "
        "split-preflight, schema-gate)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Issue #2502 P0 corpus build.")
    p.add_argument(
        "--out-dir",
        default="/workspace/issue2502_corpus",
        help="Output dir for corpus.jsonl + reports + staged/ checkpoints.",
    )
    p.add_argument("--budget", type=int, default=BUDGET, help="Total context budget (150k).")
    p.add_argument("--seed", type=int, default=SEED, help="Subsample seed.")
    p.add_argument(
        "--stream-cap",
        type=int,
        default=None,
        help="Cap streamed rows per (source,config) — tiny smoke only.",
    )
    p.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Subset of source_tag(s) to build (default: all 12 families).",
    )
    p.add_argument(
        "--tokenizers",
        nargs="+",
        default=list(DEFAULT_TOKENIZERS),
        help="ALL model tokenizers for the MAX-context-token filter; a kept "
        "row must fit the budget under EVERY one (cross-model common-corpus "
        "contract — default: both Model A and Model B).",
    )
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument("--gen-headroom", type=int, default=GEN_HEADROOM_TOKENS)
    p.add_argument("--token-margin", type=int, default=TOKEN_MARGIN)
    p.add_argument(
        "--no-token-filter",
        action="store_true",
        help="Disable the MAX-context-token filter (char-length filter still runs).",
    )
    p.add_argument(
        "--no-language-filter",
        action="store_true",
        help="Disable the English-only filter on WildChat/LMSYS.",
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="Counts-only: realized per-source post-dedup yields + re-scaled "
        "budget allocation; no full corpus write.",
    )
    p.add_argument("--upload", action="store_true", help="Upload the corpus + reports to HF.")
    p.add_argument(
        "--upload-repo",
        default=hub.DEFAULT_DATASET_REPO,
        help="HF dataset repo for the corpus upload.",
    )
    p.add_argument(
        "--upload-prefix",
        default="issue2502_ctxmap_xgen/context_corpus",
        help="HF path_in_repo prefix for the corpus + reports.",
    )
    p.add_argument(
        "--skip-split-preflight",
        action="store_true",
        help="Skip the registry-wide declared-(config,split) existence preflight "
        "(offline fixture runs only; a falsely-failing preflight can be bypassed "
        "after manual verification — gotchas.md preflight-gate escape hatch).",
    )
    p.add_argument(
        "--skip-schema-gate",
        action="store_true",
        help="Skip the round-10 per-source tiny-real streaming schema gate "
        "(offline fixture runs only; same escape-hatch contract as "
        "--skip-split-preflight).",
    )
    p.add_argument(
        "--schema-gate-rows",
        type=int,
        default=SCHEMA_GATE_ROWS,
        help="Rows streamed per (source, config, split[, file]) attempt by the "
        "schema gate (bounded tiny-real probe).",
    )
    p.add_argument(
        "--schema-gate-only",
        action="store_true",
        help="Run the split preflight + schema gate against the real registry, "
        "write schema_gate_report.json, and exit (no staging) — the pre-launch "
        "coverage run.",
    )
    p.add_argument(
        "--import-check",
        action="store_true",
        help="Argparse-attribute + helper-call-bind check, then exit (no work).",
    )
    p.add_argument(
        "--selfcheck",
        action="store_true",
        help="Synthetic-fixture self-check of the pure data-plane functions "
        "(no network, no tokenizer download), then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_dotenv()  # shared-VM thread caps + HF_TOKEN before any heavy import
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_corpus: import-check OK")
        return 0
    if args.selfcheck:
        run_selfcheck()
        return 0
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
