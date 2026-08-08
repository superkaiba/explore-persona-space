#!/usr/bin/env python
"""Issue #1773 — shared constants + pure helpers for the SAE feature description
+ categorization pipeline (plan v5).

Holds everything more than one issue1773 stage imports: paths/seeds, evidence
design constants (plan §11), the five judged-axis label sets + rubrics
(speaker_property ADOPTED VERBATIM from #1092 v13 SPEAKER_CLASSES), the
deterministic per-draw label permutation (cache-key differentiation, llm-judging
rule 22), majority-vote + drop semantics (rule 9/24 split), varying-n Fleiss
kappa, the registered §3 verdict lattice, and evidence-packet rendering.

Pure functions only (no model / store / API access) so tests import it cheaply.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

TASK_ID = 1773
SEED = 17_732_026
LAYER = 19
ACT_DIM = 3584
DICT_SIZE = 131_072
N_SHARDS = 1920

# ── canonical paths ──────────────────────────────────────────────────────────
STORE_DEFAULT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1482_shuffnull/"
    "issue1482_error_analysis/analysis_tensors/sae_pooled"
)
PERFEATURE_NPZ = PROJECT_ROOT / "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
OUT_EVAL = PROJECT_ROOT / "eval_results/issue_1773"
OUT_FIGS = PROJECT_ROOT / "figures/issue_1773"
WORK_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1773_evidence")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1773_featurepipeline"
HF_SELECTION_PREFIX = f"{HF_PREFIX}/selection"
# Canonical Pass-A selection dir (crash-fix r3 path unification, #1773): ONE
# repo-relative constant every consumer (pass_select / pass_windows / the Pass-B
# launcher) reads, overridable via EPM_1773_SEL_DIR. Reconciles the plan's
# `data/issue_1773/evidence/selection/` prose, the launcher's
# `$REPO_ROOT/data/issue_1773/selection`, and the old argparse default
# WORK_DEFAULT/selection (where the 2026-07-28 VM Pass-A run actually wrote —
# point EPM_1773_SEL_DIR there to consume that copy).
SEL_DIR_DEFAULT = Path(
    os.environ.get("EPM_1773_SEL_DIR", str(PROJECT_ROOT / "data" / "issue_1773" / "selection"))
)

# ── evidence design (plan §11, Sources: arXiv 2410.13928 / Delphi / 2605.12874) ──
N_ACT_BINS = 10  # quantile bins over the feature's own ans_max distribution
ACT_PER_BIN = 6  # 4 evidence + 2 held-out per bin
ACT_EVIDENCE_PER_BIN = 4
N_ACT_EVIDENCE = N_ACT_BINS * ACT_EVIDENCE_PER_BIN  # 40
N_ACT_HOLDOUT = N_ACT_BINS * (ACT_PER_BIN - ACT_EVIDENCE_PER_BIN)  # 20
N_NONACT_EVIDENCE = 20
N_NONACT_HOLDOUT = 6
N_NONACT = N_NONACT_EVIDENCE + N_NONACT_HOLDOUT  # 26
NONACT_DRAW_BUFFER = 96  # oversample; rejection-filtered against active rows
N_NEAR_MISS = 5  # from top-3 decoder-cosine neighbours' top-bin evidence windows
N_NEIGHBORS = 8  # phase0 neighbours table (top-8 decoder cosine)
WINDOW_BACK = 15  # window = [peak-15 .. peak+16] (32 tokens), Source: 2410.13928
WINDOW_FWD = 16
N_RANDOM_DIRECTIONS = 200  # random-direction control features (plan §5)
RAND_TOP_K = 66  # bounded top-K (row, dot-max) kept per random direction
RESERVOIR_PER_FEATURE = 660  # uniform reservoir ≈66/decile in expectation (Pass A)

# ── judged axes (plan §4 Phase 3) ────────────────────────────────────────────
# speaker_property labels ADOPTED VERBATIM from #1092 v13 SPEAKER_CLASSES
# (worktree issue-1092-crossed-core-sae, scripts/issue1092_crossed_core_sae.py:140):
#   ("language", "register_style", "identity_disposition", "none", "unclear")
AXES: dict[str, tuple[str, ...]] = {
    "abstraction": ("token_surface", "lexical_semantic", "abstract_contextual"),
    "speaker_property": ("language", "register_style", "identity_disposition", "none", "unclear"),
    "content_type": ("topic", "task_format", "entity", "syntax", "operation"),
    "functional_role": ("input_side", "output_promoting", "mixed"),
    "interpretable": ("yes", "no"),
}
N_DRAWS = 5  # llm-judging rule 4; the persona_related kappa 0.136@N=1 failure this fixes
MAJORITY_FLOOR = 3  # majority vote needs >=3 agreeing surviving draws, else `unresolved`
AXES_MAX_TOKENS = 400  # llm-judging rule 23 (reason-then-label rubric >= ~300)
DESCRIBE_MAX_TOKENS = 700  # free-text description, not length-capped to a word
JUDGE_TEMPERATURE = 1.0

# Per-axis label DEFINITIONS (sharp definitions — 2506.13639; speaker_property
# definition text verbatim from #1092's SPEAKER_JUDGE_SYSTEM, reshaped one-per-line).
AXIS_DEFINITIONS: dict[str, dict[str, str]] = {
    "abstraction": {
        "token_surface": (
            "the feature fires on a specific token, string, punctuation mark, or surface "
            "pattern regardless of meaning"
        ),
        "lexical_semantic": (
            "the feature fires on a word/phrase meaning or a narrow semantic field "
            "(synonyms, inflections of one concept)"
        ),
        "abstract_contextual": (
            "the feature fires on an abstract, contextual, or discourse-level property "
            "spanning many unrelated surface forms"
        ),
    },
    "speaker_property": {
        "language": (
            "the shared property is which natural language / script the answers are written in"
        ),
        "register_style": "the formality, tone, genre, or verbosity register of the answers",
        "identity_disposition": (
            "who the speaker IS or a stable trait/stance of the speaker - "
            "self-identification, refusal disposition, sycophancy, persona compliance, "
            "first-person identity"
        ),
        "none": (
            "the shared property is topical content, task type, formatting, markup, or "
            "code syntax - nothing about the speaker"
        ),
        "unclear": "no coherent shared property is discernible from the examples",
    },
    "content_type": {
        "topic": "a subject-matter domain (e.g. medicine, sports, cooking)",
        "task_format": "a kind of task or output format (e.g. lists, translations, Q&A shape)",
        "entity": "specific named entities (people, places, products, organizations)",
        "syntax": "a grammatical or syntactic structure independent of topic",
        "operation": (
            "an operation the text performs (negation, enumeration, comparison, a code operation)"
        ),
    },
    "functional_role": {
        "input_side": (
            "the feature tracks a property of the input/context it reads; its promoted "
            "output tokens look incidental"
        ),
        "output_promoting": (
            "the feature drives specific next-token outputs (its promoted-token list "
            "matches what follows the marked tokens)"
        ),
        "mixed": "clear evidence of both input-tracking and output-promotion",
    },
    "interpretable": {
        "yes": "a coherent shared property of the marked tokens/examples is identifiable",
        "no": "no coherent shared property is identifiable from the examples",
    },
}

DESCRIBER_SYSTEM = (
    "You are labeling features of a sparse autoencoder trained on a language model's "
    "internal activations. Each feature activates on specific tokens in text. You will "
    "see example texts where the feature fires (the strongest token marked <<like "
    "this>>), example texts where it does NOT fire, and the output tokens the feature "
    "promotes/suppresses. Describe the shared property of the delimited tokens and "
    "their contexts. Reason briefly, then output ONLY JSON: "
    '{"reasoning": "...", "description": "<one or two sentences naming the shared '
    'property>", "confidence": <integer 0-100>}. The description must be specific '
    "enough that a reader could pick out new texts that activate the feature."
)

AXIS_SYSTEM_PREAMBLE = (
    "You are categorizing one sparse-autoencoder feature of a language model from "
    "evidence (example texts with the strongest activating token marked <<like this>>, "
    "and/or a feature description). Answer ONE classification question. Reason "
    "briefly, then output ONLY JSON: "
    '{"reasoning": "...", "label": "<exactly one allowed label>"}. '
    "The label MUST be copied verbatim from the allowed set. Choose exactly ONE."
)

# What each axis SEES (plan §4 Phase 3 input matrix). R2 / arm shares / map
# outputs NEVER appear in any prompt (blinding rule 1); STAT withheld from all.
AXIS_SEES: dict[str, tuple[str, ...]] = {
    "abstraction": ("EX_POS", "EX_NEG", "DESC"),
    "speaker_property": ("EX_POS_DIVERSE", "EX_NEG", "NEAR", "DESC"),
    "content_type": ("EX_POS", "NEAR", "DESC"),
    "functional_role": ("EX_POS", "OUT", "DESC"),
    "interpretable": ("EX_POS", "EX_NEG", "DESC"),
}
# Evidence subsets in axis prompts (plan §9: axes input trimmed toward ~1.5k tokens
# where evidence subsets allow; per-call window groups 4-8 sanctioned in §13).
AXIS_EX_POS_N = 10
AXIS_EX_NEG_N = 5

CONTENT_DROP_REASONS = ("refusal", "malformed", "out_of_set")

# Consumer-facing per-axis usability marking (#1941 P4). #1941's decide-mark
# phase resolved the registered verdict lattice to RETIRE (no arm cleared the
# repair bars; eval_results/issue_1941/decision.json), so the functional_role
# entry below carries the "unusable: ..." string verbatim from
# issue1941_fr_diag.usability_string. Consumers joining against
# `feature_table_v1.jsonl` axis columns should check this constant (and the
# per-row `axis_usability` field the same phase adds) before treating a
# column as reliable.
AXIS_USABILITY: dict[str, str] = {
    "functional_role": (
        "unusable: inter-draw kappa 0.318 vs 0.63-0.71 siblings (#1941; RETIRE — "
        "no arm cleared the registered repair lattice)"
    ),
}


def label_permutation(feat_id: int, axis: str, draw_idx: int) -> list[str]:
    """Deterministic per-(feat, axis, draw) permutation of the axis label set.

    Enters the USER-message template, so the 5 draws' rubric-keyed cache keys
    differ (llm-judging rule 22) and label-order position bias is controlled
    (arXiv 2602.02219). Pure function of the inputs — no global RNG state.
    """
    labels = list(AXES[axis])
    h = hashlib.sha256(f"{TASK_ID}|{feat_id}|{axis}|{draw_idx}".encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    import numpy as np

    rng = np.random.default_rng(seed)
    return [labels[i] for i in rng.permutation(len(labels))]


def axis_custom_id(feat_id: int, axis: str, draw_idx: int) -> str:
    """Batch custom_id, <=53 chars (64-char API cap minus the 11-char encoder
    suffix — the #1415 budget). Longest: 'f131071-speaker_property-d4' = 27."""
    cid = f"f{feat_id}-{axis}-d{draw_idx}"
    assert len(cid) <= 53, cid
    return cid


def parse_axis_custom_id(cid: str) -> tuple[int, str, int]:
    """Inverse of axis_custom_id."""
    body = cid[1:]
    feat_s, axis, draw_s = body.rsplit("-", 2)
    return int(feat_s), axis, int(draw_s[1:])


def validate_axis_label(parsed: object, axis: str) -> str | None:
    """Drop-never-coerce label validation (llm-judging rule 9).

    Returns the normalized label when parsed is a dict carrying a `label`
    verbatim in (or case/space-normalizable to) the axis set; None = content
    drop (REFUSAL / malformed / out-of-set). Never coerces.
    """
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("label")
    if not isinstance(raw, str):
        return None
    norm = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return norm if norm in AXES[axis] else None


def majority_vote(labels: list[str]) -> str:
    """Majority label over SURVIVING draws with >= MAJORITY_FLOOR agreeing votes;
    ties or below-floor -> 'unresolved' (reported, never coerced)."""
    if not labels:
        return "unresolved"
    counts: dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    best = max(counts.values())
    winners = [k for k, v in counts.items() if v == best]
    if len(winners) != 1 or best < MAJORITY_FLOOR:
        return "unresolved"
    return winners[0]


def fleiss_kappa_varying_n(items: list[list[str]], categories: tuple[str, ...]) -> dict:
    """Fleiss kappa with the varying-n extension (plan §3: features with dropped
    draws keep their surviving draws; items with < 2 surviving draws are
    excluded and counted). Returns kappa + prevalence + raw agreement
    (reported NEXT TO kappa per the lattice-granularity note).
    """
    used = [it for it in items if len(it) >= 2]
    n_excluded = len(items) - len(used)
    if not used:
        return {
            "kappa": float("nan"),
            "n_items": 0,
            "n_excluded_lt2": n_excluded,
            "prevalence": {},
            "raw_agreement": float("nan"),
            "scheme": "varying-n Fleiss extension (items with <2 surviving draws excluded)",
        }
    cat_idx = {c: i for i, c in enumerate(categories)}
    p_sum = 0.0
    tot_by_cat = [0] * len(categories)
    tot_n = 0
    for it in used:
        n_i = len(it)
        counts = [0] * len(categories)
        for lab in it:
            counts[cat_idx[lab]] += 1
        p_sum += (sum(c * c for c in counts) - n_i) / (n_i * (n_i - 1))
        for j, c in enumerate(counts):
            tot_by_cat[j] += c
        tot_n += n_i
    p_bar = p_sum / len(used)
    p_j = [c / tot_n for c in tot_by_cat]
    p_e = sum(p * p for p in p_j)
    kappa = float("nan") if math.isclose(p_e, 1.0) else (p_bar - p_e) / (1.0 - p_e)
    return {
        "kappa": kappa,
        "n_items": len(used),
        "n_excluded_lt2": n_excluded,
        "prevalence": {c: p_j[i] for c, i in cat_idx.items()},
        "raw_agreement": p_bar,
        "scheme": "varying-n Fleiss extension (items with <2 surviving draws excluded)",
    }


# ── registered verdict lattice (plan §3, DISJOINT + exhaustive) ──────────────
LATTICE_DETECTION_MIN = 0.70
LATTICE_FUZZING_MIN = 0.70
LATTICE_DISCRIMINATION_MIN = 0.50
LATTICE_KAPPA_MIN = 0.60
LATTICE_SHUFFLED_MAX = 0.55


def apply_lattice(row: dict) -> str:
    """Plan §3 lattice per axis: TRUSTWORTHY iff detection >= 0.70 AND fuzzing
    >= 0.70 AND discrimination >= 0.50 AND inter-draw kappa >= 0.6 AND
    shuffled-label detection <= 0.55; SEARCH-INDEX-ONLY otherwise (a NaN in
    any conjunct fails that conjunct — never silently passes)."""

    def _ge(v: object, bar: float) -> bool:
        return isinstance(v, int | float) and not math.isnan(float(v)) and float(v) >= bar

    def _le(v: object, bar: float) -> bool:
        return isinstance(v, int | float) and not math.isnan(float(v)) and float(v) <= bar

    ok = (
        _ge(row.get("detection"), LATTICE_DETECTION_MIN)
        and _ge(row.get("fuzzing"), LATTICE_FUZZING_MIN)
        and _ge(row.get("discrimination"), LATTICE_DISCRIMINATION_MIN)
        and _ge(row.get("kappa"), LATTICE_KAPPA_MIN)
        and _le(row.get("shuffled_detection"), LATTICE_SHUFFLED_MAX)
    )
    return "TRUSTWORTHY" if ok else "SEARCH-INDEX-ONLY"


# ── evidence rendering ───────────────────────────────────────────────────────


def render_windows_block(title: str, windows: list[dict], with_marks: bool = True) -> str:
    """Render a numbered window block. Window dicts carry `text_marked` (peak
    token in <<...>>) + `text_plain`; per-token activation VALUES are omitted
    from prompts (2410.13928: +0.01 — plan §11)."""
    lines = [f"### {title}"]
    for i, w in enumerate(windows, 1):
        txt = w["text_marked"] if with_marks else w["text_plain"]
        lines.append(f"{i}. {txt}")
    return "\n".join(lines)


def render_out_block(footprint: dict) -> str:
    """Render the logit-footprint OUT block (top-10 promoted / suppressed token
    strings, values omitted)."""
    prom = ", ".join(repr(t) for t in footprint.get("top_promoted_tokens", []))
    supp = ", ".join(repr(t) for t in footprint.get("top_suppressed_tokens", []))
    return f"### Output tokens\npromoted: {prom}\nsuppressed: {supp}"


def build_describe_user_msg(packet: dict) -> str:
    """Describer user message: EX+ (40, marked) + EX- (20, plain) + OUT block."""
    parts = [
        render_windows_block(
            "Examples where the feature ACTIVATES (strongest token marked <<...>>)",
            packet["ex_pos"],
            with_marks=True,
        ),
        render_windows_block(
            "Examples where the feature does NOT activate",
            packet["ex_neg"],
            with_marks=False,
        ),
    ]
    if packet.get("out"):
        parts.append(render_out_block(packet["out"]))
    parts.append(
        "Describe the shared property of the <<marked>> tokens and their contexts. "
        "Output the JSON now."
    )
    return "\n\n".join(parts)


def _diverse_pos_subset(ex_pos: list[dict], n: int) -> list[dict]:
    """Topic-diverse EX+ draw for speaker_property: <=1 window per context (ci),
    spread across quantile bins (plan §4 Phase 3 table)."""
    seen_ci: set[int] = set()
    by_bin: dict[int, list[dict]] = {}
    for w in ex_pos:
        by_bin.setdefault(int(w.get("bin", 0)), []).append(w)
    picked: list[dict] = []
    bins = sorted(by_bin)
    while len(picked) < n and any(by_bin[b] for b in bins):
        for b in bins:
            while by_bin[b]:
                w = by_bin[b].pop(0)
                ci = int(w.get("ci", -1))
                if ci not in seen_ci:
                    seen_ci.add(ci)
                    picked.append(w)
                    break
            if len(picked) >= n:
                break
    return picked


def build_axis_user_msg(
    axis: str,
    packet: dict,
    description: str | None,
    draw_idx: int,
    *,
    sees: tuple[str, ...] | None = None,
) -> str:
    """One-axis categorization user message with the per-draw PERMUTED label
    order rendered inline (cache-key differentiation + position-bias control).

    ``sees`` (kw-only, #1941): evidence-block override for the functional_role
    evidence A/B arms. ``None`` (the default) reads ``AXIS_SEES[axis]`` —
    byte-identical to the pre-#1941 behavior (pinned by the golden-prompt
    tests in tests/test_issue1941_fr_diag.py). Block RENDER ORDER is fixed
    (EX_POS/EX_POS_DIVERSE, EX_NEG, NEAR, OUT, DESC, question) regardless of
    tuple order.
    """
    feat_id = int(packet["feat_id"])
    sees = AXIS_SEES[axis] if sees is None else tuple(sees)
    parts: list[str] = []
    if "EX_POS_DIVERSE" in sees:
        parts.append(
            render_windows_block(
                "Activating examples (strongest token marked; one per conversation)",
                _diverse_pos_subset(packet["ex_pos"], AXIS_EX_POS_N),
                with_marks=True,
            )
        )
    elif "EX_POS" in sees:
        parts.append(
            render_windows_block(
                "Activating examples (strongest token marked <<...>>)",
                packet["ex_pos"][:AXIS_EX_POS_N],
                with_marks=True,
            )
        )
    if "EX_NEG" in sees:
        parts.append(
            render_windows_block(
                "Non-activating examples", packet["ex_neg"][:AXIS_EX_NEG_N], with_marks=False
            )
        )
    if "NEAR" in sees and packet.get("near"):
        parts.append(
            render_windows_block(
                "NEAR-MISS examples (a similar but DIFFERENT feature activates here)",
                packet["near"],
                with_marks=True,
            )
        )
    if "OUT" in sees and packet.get("out"):
        parts.append(render_out_block(packet["out"]))
    if "DESC" in sees and description:
        parts.append(f"### Feature description\n{description}")
    perm = label_permutation(feat_id, axis, draw_idx)
    defs = AXIS_DEFINITIONS[axis]
    opt_lines = [f"- {lab}: {defs[lab]}" for lab in perm]
    parts.append(
        "### Question\n"
        f"Classify this feature on the `{axis}` axis. Allowed labels (choose exactly one):\n"
        + "\n".join(opt_lines)
        + '\n\nOutput ONLY JSON: {"reasoning": "...", "label": "<one allowed label>"}.'
    )
    return "\n\n".join(parts)


def sha16(text: str) -> str:
    """16-hex-char sha256 prefix (prompt-hash reporting)."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def repro_meta() -> dict:
    """Reproducibility metadata for result JSONs (commit, versions, ts)."""
    import datetime
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    commit = out.stdout.strip() if out.returncode == 0 else "unknown"
    import numpy

    return {
        "git_commit": commit,
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "task": TASK_ID,
        "seed": SEED,
    }


def write_jsonl_sharded(rows: list[dict], out_dir: Path, stem: str, max_bytes: int = 9_000_000):
    """Write rows as `<stem>.shardNN.jsonl` files each < max_bytes (<9 MB, never
    gzip — upload policy), plus a `<stem>.manifest.json`. Returns shard paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    buf: list[str] = []
    size = 0

    def _flush():
        nonlocal buf, size
        if not buf:
            return
        p = out_dir / f"{stem}.shard{len(shards):02d}.jsonl"
        tmp = p.parent / f".tmp_{p.name}"
        tmp.write_text("\n".join(buf) + "\n")
        tmp.replace(p)
        shards.append(p)
        buf, size = [], 0

    for r in rows:
        line = json.dumps(r, ensure_ascii=False)
        if size + len(line.encode()) + 1 > max_bytes:
            _flush()
        buf.append(line)
        size += len(line.encode()) + 1
    _flush()
    manifest = {
        "stem": stem,
        "n_rows": len(rows),
        "shards": [p.name for p in shards],
        **repro_meta(),
    }
    (out_dir / f"{stem}.manifest.json").write_text(json.dumps(manifest, indent=1))
    return shards


def iter_jsonl(path: Path):
    """Text-mode JSONL iteration (never splitlines() — U+2028/NEL shred rule)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
