#!/usr/bin/env python
"""P2 VM-side Batch-API judge-wave driver for task #2552 (unit 2 of the 3-unit build).

Implements plan v4 §4 P2 + §7 G2/G3 + MF-A/MF-D over unit 1's banked artifacts:

- prep      — pinned input fetch (HF ``issue2552_turnsae/`` per-file, never snapshot),
              the G2 measured-union descope gate (total description-need union across the
              4 families > 45,000 => eval turns 2,000 -> 1,000 AND rep panel 12,000 -> 8,000,
              recomputed once), the MF-A eval/mining disjointness HARD-ASSERT, and the
              checkpointed eval-turn text fetch (3.22 GB rollout-chunk sweep, resumable).
- w1        — per-feature descriptions (4 families; JSON ``{"description": ...}``).
- w2        — per-turn structured summaries (24-field/5-category schema VERBATIM from
              Der et al. arXiv 2606.28548 Appendix D; flat 24-key JSON).
- w3        — schema-field assignment per PANEL feature, 3 turn-averaged dictionaries
              (reason-then-label JSON ``{"reason","field","category"}``; ``none``/malformed
              DROPPED never coerced, drop counts per dictionary).
- w4        — 10-way matching per (turn x config); fixed per-turn distractor sets
              (seed 2552) SHARED across configs; validity-flagged rows to
              ``eval_results/issue_2552/dere_repl/matching_perturn.json``.
- w5        — pairwise coverage (10 config pairs x turns; presentation order seed 2552;
              equal-length lists via truncate-both-to-min) ->
              ``dere_repl/pairwise_perturn.json``.
- w6        — 5-way ranking per turn (labels A-E, seeded assignment).
- w7        — calibration re-judge: 200 W3 + 200 W4 + 200 W5 items re-judged with
              ``claude-sonnet-4-5-20250929`` (same prompts, no prefill); raw agreement +
              Cohen's kappa per instrument.
- pilot-w{1,3,4,5} — MF-D rule-26 pilots (>=51 effective draws/arm on the wave's OWN
              batch route, FRESH cache, zero-truncation + per-arm parse-fail < 2% +
              per-arm api-refusal < 0.10 gates). W2/W6 (2,000 calls each) sit below the
              ~5,000-call rule-26 floor and are exempt from the formal gate (the smoke's
              5-call live probes + post-hoc drop/completeness reports bind there).

Judge client: every call goes through ``eval.batch_judge.judge_completions_batch``
(routed via ``llm/api_dispatch.py``), ``judge_model="claude-sonnet-4-6"`` passed
explicitly per call site, NO assistant prefill anywhere, rubric-keyed ``JudgeCache``,
``threshold_base=0`` (pins the Batch route for every wave — no OTPM-probe region).

PILOT NOTE (addressed-differently vs plan prose): the plan names
``eval.judge_pilot.judge_pilot_gate`` as the pilot vehicle. That helper dispatches via
``judge_graded``, which HARD-APPENDS a graded ``{"score": 0-100}`` JSON contract to the
judge SYSTEM prompt (``graded_judge._JSON_WRAPPER``) and reduces via
``_score_from_parsed`` — for these NON-graded instruments the pilot would then run a
DIFFERENT instrument than the production wave (violating rule 26(c)'s exact-instrument
requirement) and read ~100% parse-fail. This driver therefore implements the rule-26
clauses (a)-(d) itself — reference implementation
``eval.judge_pilot.judge_pilot_gate`` — via the SAME ``judge_completions_batch`` call
shape as production: fresh pilot cache (``n_cached == 0`` FAILs), realized route read
back from ``save_raw["routing"]`` (must be ``path == "batch"``), zero
``stop_reason == "max_tokens"``, per-arm parse-fail < 2% under the wave's OWN parser,
per-arm api-refusal < 0.10, >=51 effective draws per arm (#2124 floor at the 2%
threshold). Divergence from the reference: the graded score wrapper + graded reduce are
dropped (instrument-mismatched); every other clause is mirrored.

Rule-28 remediation: censored items (api-refusal / residual-transport draws) are
re-issued on the SYNC path at the IDENTICAL instrument against the SAME cache dir
(censor-class dicts are cache put-SKIPPED, so they re-dispatch), merged with surviving
batch draws, with the batch/sync split disclosed in ``judge_meta_<wave>.json``.
Per-item ``frac_items_complete`` is reported per wave/arm against the pre-registered
0.95 floor (rule 29).

Raw persistence: per-draw JSONL (full raw text incl. rationales + stop_reason) is
sharded (<9 MB line-shards, never gzip) and uploaded to HF
``issue2552_turnsae/raw_completions/judge/<wave>/`` BEFORE any reduction.

Content hygiene: LMSYS/WildChat turn/rollout text is DIGEST-ONLY in every log line
(ids + counts + shas; text fields are never printed).

CLI: ``--wave {prep,w1..w7,pilot-w1,pilot-w3,pilot-w4,pilot-w5,all}`` + ``--dry-run``
(compose + zero-API routing check) + ``--smoke`` (5-item live SYNC probe per
instrument, outputs diverted under the work root) + ``--import-check`` /
``--list-phases``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before any torch-importing lazy module; credentials + VM thread caps

import argparse  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from explore_persona_space.eval.batch_judge import (  # noqa: E402
    is_api_refusal_error_dict,
    is_api_refusal_stop_reason,
    is_transport_error_dict,
    is_truncation_error_dict,
    judge_completions_batch,
)
from explore_persona_space.eval.judge_dispatch import keep_raw_judge_text  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2552.judge")

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── instrument constants (plan §4 P2 / §11) ─────────────────────────────────────
JUDGE_MODEL = "claude-sonnet-4-6"  # plan §11 (A8 probe-verified); passed explicitly per call
CAL_JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # W7 calibration anchor (project judge)
WAVE_THRESHOLD_BASE = 0  # pins the Batch route for EVERY wave (no OTPM-probe region)
SEED = 2552
G2_UNION_CAP = 45_000
G2_EVAL_N = 1_000
G2_PANEL_CAP = 8_000
FRAC_ITEMS_FLOOR = 0.95  # rule-29 pre-registered per-arm completeness floor
MAX_TOKENS = {"w1": 1024, "w2": 2048, "w3": 1024, "w4": 1024, "w5": 1024, "w6": 1024}
W7_MAX_TOKENS = 1024  # parents W3/W4/W5 all run at 1024
W7_N_PER_INSTRUMENT = 200
EVAL_TURN_TEXT_CAP = 4_000  # chars: the summarized turn (W2) — single-turn prompt budget
W4_CANDIDATE_CAP = 1_500  # chars per W4 candidate (10 per prompt; unit-1 example-cap parity)
W3_EXCERPTS = 3  # top-3 mining excerpts per W3 item (plan §4 P2 W3)
W2_MIN_FIELDS = 20  # W2 validity floor: >=20 of 24 fields present + non-empty
SMOKE_N_EVAL_TEXTS = 40  # --smoke: eval-text fetch + composition bound (chunk sweep stays tiny)
PILOT_MIN_EFFECTIVE = 51  # floor(1/0.02)+1 (#2124) — n_draws=1 => 51 items/arm
PILOT_PARSE_FAIL_THRESHOLD = 0.02
PILOT_API_REFUSAL_THRESHOLD = 0.10
RC_PILOT_FAIL = 7  # designed halt rc (the #1415 pilot-gate routing convention)
RC_FLOOR_FAIL = 8  # rule-29 frac_items_complete hard gate (aggregation HALT, #2552 r2)
TA_FAMILIES = ("rep_ta", "mat_k100", "mat_k200")
ALL_FAMILIES = (*TA_FAMILIES, "pt")
CONFIGS = ("rep_ta", "mat_k100", "mat_k200", "pt_max", "pt_sum")
CONFIG_FAMILY = {
    "rep_ta": "rep_ta",
    "mat_k100": "mat_k100",
    "mat_k200": "mat_k200",
    "pt_max": "pt",
    "pt_sum": "pt",
}
W5_PAIRS = tuple(itertools.combinations(CONFIGS, 2))  # 10 unordered config pairs
W4_LABELS = tuple("ABCDEFGHIJ")
W6_LABELS = tuple("ABCDE")
PILOTED_WAVES = ("w1", "w3", "w4", "w5")  # >= ~5k-call waves (rule 26); w2/w6 exempt
_ITEM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,53}$")

# Committed unit-1 / parent-#2476 inputs (git; branch issue-2552).
REGIME_JSON = PROJECT_ROOT / "eval_results" / "issue_2552" / "regime.json"
UNION_C_NPZ = (
    PROJECT_ROOT / "eval_results" / "issue_2476" / "floor_sweep" / ("perfeature_union_c.npz")
)
UNION_K200_NPZ = (
    PROJECT_ROOT / "eval_results" / "issue_2476" / "k200_census" / ("perfeature_union_k200.npz")
)

# ── Appendix D schema (VERBATIM) ────────────────────────────────────────────────
# Verbatim from Der et al., arXiv 2606.28548, Appendix D "Structured Summary Schema"
# (HTML v1, https://arxiv.org/html/2606.28548v1), fetched via WebFetch on 2026-08-24 and
# cross-checked by a second independent extraction of six spot fields (descriptions
# byte-identical across the two fetches). The appendix formats entries as a table
# (Field | Description | Example value); the example-value column is not reproduced —
# the field names + descriptions are the judged instrument. Appendix introduction:
# "We asked Sonnet 4.6 to design a structured summary schema, which yielded 24 fields
# across 5 categ[ories]".
APP_D_SCHEMA: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Content",
        (
            ("domain", "Broad subject area"),
            ("topic", "Specific subject described abstractly"),
            ("factuality", "How factual vs. opinionated, speculative, or fictional"),
            ("concreteness", "How abstract/theoretical vs. concrete/practical"),
            ("quantitative content", "Role of numbers, data, measurements — or their absence"),
            (
                "temporality",
                "Time orientation (historical, contemporary, forward-looking, timeless)",
            ),
        ),
    ),
    (
        "Form",
        (
            ("text type", "What kind of text (prose, tutorial, code, creative piece, etc.)"),
            ("language", "Primary language and any mixing"),
            ("structure", "How the response is organized"),
            ("linguistic sophistication", "Vocabulary complexity, sentence structure, formality"),
        ),
    ),
    (
        "Voice",
        (
            ("tone", "Attitude toward the reader"),
            ("emotional engagement", "Empathy, validation, encouragement vs. detachment"),
            ("persona", "Role the assistant is playing"),
            ("perspectivity", "Point of view and how the speaker positions themselves"),
            ("certainty", "How confident or hedged the claims are"),
            ("valence", "Attitude toward the subject matter itself"),
        ),
    ),
    (
        "Function",
        (
            ("intent", "What the response is trying to accomplish"),
            ("audience level", "Who this seems written for"),
            ("rhetorical strategy", "Main technique for conveying the message"),
            ("scope and depth", "How much ground is covered and at what detail"),
            ("interactivity", "Whether it invites further dialogue or stands alone"),
        ),
    ),
    (
        "Meta",
        (
            ("contextuality", "How much it depends on or references prior conversation"),
            ("epistemic signals", "Disclaimers, caveats, safety language, or their absence"),
            ("creativity", "How formulaic vs. novel the approach is"),
        ),
    ),
)
APP_D_FIELDS: tuple[str, ...] = tuple(f for _c, fs in APP_D_SCHEMA for f, _d in fs)
FIELD_TO_CATEGORY: dict[str, str] = {f: c for c, fs in APP_D_SCHEMA for f, _d in fs}
assert len(APP_D_FIELDS) == 24 and len(APP_D_SCHEMA) == 5, (
    len(APP_D_FIELDS),
    len(APP_D_SCHEMA),
)


def _schema_block() -> str:
    """Render the App-D schema as 'Category — field: description' lines."""
    lines: list[str] = []
    for cat, fields in APP_D_SCHEMA:
        lines.append(f"{cat}:")
        for f, d in fields:
            lines.append(f"  - {f}: {d}")
    return "\n".join(lines)


# ── wave system prompts (the judged instruments; NO assistant prefill anywhere) ──
_JSON_ONLY = "Output ONLY a single JSON object and nothing else."

W1_SYSTEM_TA = (
    "You are labeling features of a sparse autoencoder trained on turn-averaged language-model"
    " activations. You will be shown the top-activating conversation turns for ONE feature,"
    " each with its activation value. Write a short description (one or two sentences) of what"
    " the feature captures — the common content, style, or function of the turns it activates"
    " on. " + _JSON_ONLY + ' Use the form {"description": "<your description>"}.'
)
W1_SYSTEM_PT = (
    "You are labeling features of a sparse autoencoder trained on per-token language-model"
    " activations. You will be shown the top-activating token contexts for ONE feature: a text"
    " window around the peak token, the peak activation value, the peak token's offset within"
    " the window, and other activating token offsets in that window. Write a short description"
    " (one or two sentences) of what the feature captures. "
    + _JSON_ONLY
    + ' Use the form {"description": "<your description>"}.'
)
W2_SYSTEM = (
    "You will be shown one assistant conversation turn. Produce a structured summary of it"
    " using EXACTLY this schema — populate each of the 24 fields with a short description of"
    " THIS turn (a phrase or one sentence per field); do not quote long spans of the turn."
    "\n\nSCHEMA (5 categories, 24 fields):\n"
    + _schema_block()
    + "\n\n"
    + _JSON_ONLY
    + " Use a single flat JSON object whose keys are exactly the 24 field names above and"
    " whose values are short strings."
)
W3_SYSTEM = (
    "You classify sparse-autoencoder feature descriptions against a fixed structured-summary"
    " schema.\n\nSCHEMA (5 categories, 24 fields):\n"
    + _schema_block()
    + "\n\nYou will be shown one feature's description and up to three example excerpts from"
    " its top-activating text. Decide which ONE of the 24 fields above best names the axis"
    " this feature varies along, or 'none' if no field fits. Reason briefly first, then"
    " answer. "
    + _JSON_ONLY
    + ' Use the form {"reason": "<brief reasoning>", "field": "<one of the 24 field names,'
    ' or none>", "category": "<that field\'s category, or none>"}.'
)
W4_SYSTEM = (
    "You will be shown a list of feature descriptions that were derived from ONE assistant"
    " conversation turn, followed by 10 candidate turns labeled A through J. Exactly one"
    " candidate is the turn the descriptions were derived from. Decide which. Reason briefly"
    " first, then answer. "
    + _JSON_ONLY
    + ' Use the form {"reason": "<brief reasoning>", "choice": "<one letter A-J>"}.'
)
W5_SYSTEM = (
    "You will be shown a structured summary of one assistant conversation turn, followed by"
    " two lists of feature descriptions (List 1 and List 2) derived from that same turn by two"
    " different methods. Decide which list more completely covers what the summary says about"
    " the turn. Reason briefly first, then answer. "
    + _JSON_ONLY
    + ' Use the form {"reason": "<brief reasoning>", "choice": 1} or'
    ' {"reason": "<brief reasoning>", "choice": 2}.'
)
W6_SYSTEM = (
    "You will be shown a structured summary of one assistant conversation turn, followed by"
    " five lists of feature descriptions labeled A through E, each derived from that same turn"
    " by a different method. Rank ALL FIVE lists from best to worst coverage of what the"
    " summary says about the turn. Reason briefly first, then answer. "
    + _JSON_ONLY
    + ' Use the form {"reason": "<brief reasoning>", "ranking": ["<best label>", ...,'
    ' "<worst label>"]} listing all five labels exactly once.'
)
WAVE_SYSTEMS = {
    "w1_ta": W1_SYSTEM_TA,
    "w1_pt": W1_SYSTEM_PT,
    "w2": W2_SYSTEM,
    "w3": W3_SYSTEM,
    "w4": W4_SYSTEM,
    "w5": W5_SYSTEM,
    "w6": W6_SYSTEM,
}

PILOT_WAIVE_PARSE_FAIL_REASON = (
    "reference judge_pilot_gate's graded 0-100 reduce (_score_from_parsed) is"
    " instrument-mismatched for these non-graded waves; the REAL per-arm parse-fail"
    " (< 2%) is computed by this driver's own wave parser over the same pilot draws"
    " and gates below (module docstring PILOT NOTE)"
)


def _user_msg(question: str, completion: str) -> str:
    """format_user_msg for every wave: the composed content block IS the user message.

    The wave instructions + output contract live in the SYSTEM prompt, so the rubric
    identity (rubric_fingerprint = judge model + system + this template rendered on
    sentinels) is wave-distinct via the system prompt. ``completion`` is unused by
    design (n_draws=1, answer slot empty).
    """
    del completion
    return question


# ── lazy heavy imports (torch via unit 1; executed by --import-check) ────────────


@functools.lru_cache(maxsize=1)
def _t2552():
    """Deferred import of unit 1 (transitively torch/vendored SAE modules)."""
    import issue2552_turnsae_der as t2552

    return t2552


def _sha_ids(ids: np.ndarray) -> str:
    return _t2552()._sha_ids(np.asarray(ids, np.int64))


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── work-root layout ─────────────────────────────────────────────────────────────


def _paths(args) -> SimpleNamespace:
    """Resolve all output roots. Under --smoke EVERY output (aggregates included)
    diverts under <out_root>/smoke — canonical committed paths are never written."""
    work = Path(args.out_root)
    if args.smoke:
        work = work / "smoke"
        agg = work / "judge_aggregates"
        dere = work / "dere_repl"
    else:
        agg = PROJECT_ROOT / "eval_results" / "issue_2552" / "judge_aggregates"
        dere = PROJECT_ROOT / "eval_results" / "issue_2552" / "dere_repl"
    for d in (work, agg, dere):
        d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(work=work, agg=agg, dere=dere, inputs=work / "inputs")


# ── prep: pinned input fetch ─────────────────────────────────────────────────────


def _resolve_data_repo_revision() -> str:
    """Resolve the data repo's main -> sha ONCE per prep (revision=None on paired
    files splits across snapshot dirs — the #2061 trap)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    t = _t2552()
    api = HfApi()
    info = hub.retry_transient(
        lambda: api.repo_info(t.C.HF_DATA_REPO, repo_type="dataset"),
        what="resolve data-repo main sha",
    )
    return str(info.sha)


def _list_mining_files(revision: str, hf_prefix: str) -> list[str]:
    """Scoped listing of the mining jsonls (never full list_repo_files on the
    ~1M-file repo). Returns repo-relative paths under raw_completions/mining."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    t = _t2552()
    api = HfApi()
    prefix = f"{hf_prefix}/raw_completions/mining"
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: call runs inside hub.retry_transient (enclosing lambda)
            api.list_repo_tree(
                t.C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", revision=revision
            )
        ),
        what=f"list mining files under {prefix}",
    )
    names = [e.path for e in entries if e.path.endswith(".jsonl")]
    assert names, f"no mining jsonls found under {prefix}@{revision[:8]}"
    return sorted(names)


def phase_prep(args) -> None:
    """Fetch pinned inputs, run G2 + MF-A, fetch eval-turn texts (checkpointed)."""
    t = _t2552()
    p = _paths(args)
    p.inputs.mkdir(parents=True, exist_ok=True)
    manifest_path = p.inputs / "prep_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    revision = manifest.get("data_repo_revision") or _resolve_data_repo_revision()
    fetched: dict[str, str] = dict(manifest.get("files", {}))

    def fetch(path_in_repo: str) -> Path:
        got = t._hf_fetch(path_in_repo, p.inputs, revision)
        fetched[path_in_repo] = hashlib.sha256(Path(got).read_bytes()).hexdigest()
        return Path(got)

    hf = args.hf_prefix
    regime_measured = fetch(f"{hf}/analysis_tensors/eval/regime_measured.json")
    census_rep = fetch(f"{hf}/analysis_tensors/eval/census_rep.npz")
    # sharded lists (P1.10 writer shape, #2552 r2 g3-M4): index JSON + per-config shards
    lists_path = fetch(f"{hf}/analysis_tensors/eval_lists/feature_lists_2000turns.json")
    lists_index = json.loads(lists_path.read_text())
    assert "configs" in lists_index, (
        f"feature_lists_2000turns.json is not the sharded index (have {sorted(lists_index)}) — "
        "P1.10 writes an index + per-config JSONL turn shards beside it"
    )
    for entry in lists_index["configs"].values():
        for name in entry["files"]:
            fetch(f"{hf}/analysis_tensors/eval_lists/{name}")
    mining_manifest = fetch(f"{hf}/analysis_tensors/mining_manifest.json")
    fetch(f"{hf}/analysis_tensors/mining_row_ids.npz")  # MF-A manifest-side ids (#2552 r2 g3-M1)
    mining_files = [fetch(pth) for pth in _list_mining_files(revision, hf)]
    manifest.update(
        {
            "data_repo_revision": revision,
            "files": fetched,
            "n_mining_files": len(mining_files),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    t.C.write_json_atomic(manifest_path, manifest)
    logger.info("[prep] inputs staged at %s (revision %s)", p.inputs, revision[:12])

    # committed regime + union npzs (git; fail loud with the sparse-cone recipe)
    for pth in (REGIME_JSON, UNION_C_NPZ, UNION_K200_NPZ):
        assert pth.exists(), (
            f"committed input missing on disk: {pth} — in a sparse worktree run "
            "'git sparse-checkout add eval_results/issue_2552 eval_results/issue_2476' first"
        )

    # scratch_meta (row_ci / prov — HF-permanent, unit-1 staging helper)
    ns = SimpleNamespace(scratch=p.work / "stage")
    ns.scratch.mkdir(parents=True, exist_ok=True)
    t.EL._stage_scratch_meta(ns)
    row_ci = np.load(ns.scratch / "row_ci.npy")
    prov_u8 = np.load(ns.scratch / "prov.npy")

    g2 = _g2_gate(args, p, row_ci=row_ci, prov_u8=prov_u8)
    _mfa_disjointness(args, p, g2)
    text_ids = np.asarray(g2["eval_ids"], np.int64)
    if args.smoke:
        text_ids = text_ids[:SMOKE_N_EVAL_TEXTS]  # smoke bounds the 3.22 GB chunk sweep
    _fetch_eval_texts(args, p, text_ids, row_ci)
    logger.info(
        "[prep] done: n_eval=%d descoped=%s total_need=%d",
        g2["n_eval_realized"],
        g2["descoped"],
        g2["total_need"],
    )


# ── G2 measured-union descope gate ───────────────────────────────────────────────


def _load_lists(p, hf_prefix: str) -> dict[str, dict]:
    """Reassemble the per-config judged lists from the P1.10 sharded index.

    Producer shape (#2552 r2 g3-M4): ``feature_lists_2000turns.json`` is a small
    INDEX ({"configs": {cfg: {meta, files, n_turns}}}); one-turn-per-line JSONL
    shards (<9 MB each, never gzip — upload-policy) sit beside it. Files are staged
    by prep under ``p.inputs/<full repo path>`` (the hf_hub_download local_dir
    layout — the pre-r2 bare ``p.inputs/<name>`` read never resolved)."""
    base = p.inputs / hf_prefix / "analysis_tensors" / "eval_lists"
    index = json.loads((base / "feature_lists_2000turns.json").read_text())
    assert "configs" in index, f"unexpected lists index shape: {sorted(index)}"
    missing = [c for c in CONFIGS if c not in index["configs"]]
    assert not missing, f"feature_lists index missing configs: {missing}"
    out: dict[str, dict] = {}
    for cfg, entry in index["configs"].items():
        turns: list[dict] = []
        for name in entry["files"]:
            with (base / name).open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        turns.append(json.loads(line))
        assert len(turns) == int(entry["n_turns"]), (cfg, len(turns), entry["n_turns"])
        out[cfg] = {**entry.get("meta", {}), "turns": turns}
    return out


def _turn_lists(lists_doc: dict, cfg: str, eval_ids: set[int]) -> dict[int, list[int]]:
    """row_id -> judged_top100 feature ids for the realized eval subset."""
    out: dict[int, list[int]] = {}
    for turn in lists_doc[cfg]["turns"]:
        rid = int(turn["row_id"])
        if rid in eval_ids:
            out[rid] = [int(f) for f, _v in turn["judged_top100"]]
    return out


def _mat_panels() -> dict[str, np.ndarray]:
    return {
        "mat_k100": np.asarray(np.load(UNION_C_NPZ)["feat_ids"], np.int64),
        "mat_k200": np.asarray(np.load(UNION_K200_NPZ)["feat_ids"], np.int64),
    }


def compute_need_sets(
    eval_ids: set[int],
    lists_doc: dict,
    rep_panel: np.ndarray,
    mat_panels: dict[str, np.ndarray],
) -> dict[str, set[int]]:
    """Per-family description-need sets: panel UNION per-turn judged-list unions
    (TA families); pt = pt_max UNION pt_sum judged-list unions (no pt panel)."""
    need: dict[str, set[int]] = {}
    for fam in TA_FAMILIES:
        s = set(int(x) for x in (rep_panel if fam == "rep_ta" else mat_panels[fam]))
        for feats in _turn_lists(lists_doc, fam, eval_ids).values():
            s.update(feats)
        need[fam] = s
    pt: set[int] = set()
    for cfg in ("pt_max", "pt_sum"):
        for feats in _turn_lists(lists_doc, cfg, eval_ids).values():
            pt.update(feats)
    need["pt"] = pt
    return need


def stratified_eval_subset(eval_ids: np.ndarray, prov_u8: np.ndarray, n: int) -> np.ndarray:
    """Corpus-stratified (largest-share rounding) seed-2552 subset of the eval ids —
    mirrors unit 1's phase_select_eval arithmetic on the eval-id pool."""
    eval_ids = np.asarray(eval_ids, np.int64)
    lm = eval_ids[prov_u8[eval_ids] == 0]
    wc = eval_ids[prov_u8[eval_ids] == 1]
    n = min(n, len(eval_ids))
    n_lm = int(round(n * len(lm) / max(1, len(lm) + len(wc))))
    n_lm = min(max(n_lm, n - len(wc)), len(lm), n)
    n_wc = n - n_lm
    rng = np.random.default_rng(SEED)
    pick_lm = rng.choice(np.sort(lm), size=n_lm, replace=False) if n_lm else np.empty(0, np.int64)
    pick_wc = rng.choice(np.sort(wc), size=n_wc, replace=False) if n_wc else np.empty(0, np.int64)
    return np.sort(np.concatenate([pick_lm, pick_wc])).astype(np.int64)


def _g2_gate(args, p, *, row_ci: np.ndarray, prov_u8: np.ndarray) -> dict:
    """G2: total description-need union > 45,000 => eval 2,000 -> 1,000 AND rep panel
    12,000 -> 8,000; recompute ONCE; persist the decision + realized regime."""
    t = _t2552()
    out_path = p.agg / "g2_decision.json"
    regime = json.loads(REGIME_JSON.read_text())
    orig_ids = np.asarray(regime["eval_ids"], np.int64)
    assert _sha_ids(orig_ids) == regime["eval_ids_sha256"], "regime.json eval_ids sha drift"
    if out_path.exists():
        g2 = json.loads(out_path.read_text())
        if g2.get("orig_eval_ids_sha256") == regime["eval_ids_sha256"]:
            logger.info("[g2] resume: decision present (descoped=%s)", g2["descoped"])
            return g2
        raise AssertionError("g2_decision.json exists but keys drifted vs regime.json")

    census = np.load(p.inputs / args.hf_prefix / "analysis_tensors/eval/census_rep.npz")
    rep_panel = np.asarray(census["panel_ids"], np.int64)
    lists_doc = _load_lists(p, args.hf_prefix)
    mats = _mat_panels()
    measured = json.loads(
        (p.inputs / args.hf_prefix / "analysis_tensors/eval/regime_measured.json").read_text()
    )

    need = compute_need_sets(set(int(x) for x in orig_ids), lists_doc, rep_panel, mats)
    total = sum(len(s) for s in need.values())
    descoped = total > G2_UNION_CAP
    eval_ids = orig_ids
    if descoped:
        eval_ids = stratified_eval_subset(orig_ids, prov_u8, G2_EVAL_N)
        counts = np.asarray(census["counts"])
        rep_panel, _pdoc = t._activity_stratified_panel(
            counts, int(census["n_fit_rows"]), G2_PANEL_CAP, SEED
        )
        rep_panel = np.asarray(rep_panel, np.int64)
        need = compute_need_sets(set(int(x) for x in eval_ids), lists_doc, rep_panel, mats)
        logger.warning(
            "[g2] DESCOPE fired: total=%d > %d — eval %d->%d, rep panel ->%d (recomputed once)",
            total,
            G2_UNION_CAP,
            len(orig_ids),
            len(eval_ids),
            len(rep_panel),
        )
    g2 = {
        "descoped": bool(descoped),
        "union_cap": G2_UNION_CAP,
        "total_need_at_full_scale": int(total),
        "total_need": int(sum(len(s) for s in need.values())),
        "need_sizes": {f: len(s) for f, s in need.items()},
        "regime_measured_unions": {
            "eval_list_unions": measured.get("eval_list_unions"),
            "pt_desc_union_n": measured.get("pt_desc_union_n"),
            "rep_panel_n": measured.get("rep_panel_n"),
        },
        "n_eval_orig": int(len(orig_ids)),
        "n_eval_realized": int(len(eval_ids)),
        "eval_ids": [int(x) for x in eval_ids],
        "eval_ids_sha256": _sha_ids(eval_ids),
        "orig_eval_ids_sha256": regime["eval_ids_sha256"],
        "rep_panel_ids": [int(x) for x in rep_panel],
        "rep_panel_sha256": _sha_ids(rep_panel),
        "rep_panel_n_realized": int(len(rep_panel)),
        "mat_panel_sizes": {k: int(len(v)) for k, v in mats.items()},
        "seed": SEED,
        **as_metadata_dict(git_provenance(), phase="judge-g2"),
    }
    t.C.write_json_atomic(out_path, g2)
    logger.info("[g2] total_need=%d descoped=%s -> %s", total, descoped, out_path)
    return g2


# ── MF-A: eval/mining disjointness hard-assert ───────────────────────────────────


def _mining_records(p, hf_prefix: str, family: str) -> list[dict]:
    """Load a family's top-25 mining records from the staged jsonl shards."""
    base = p.inputs / hf_prefix / "raw_completions" / "mining"
    files = sorted(base.glob(f"top25_{family}.jsonl")) + sorted(
        base.glob(f"top25_{family}.shard*.jsonl")
    )
    assert files, f"no mining jsonls staged for family {family} under {base}"
    recs: list[dict] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def _mfa_disjointness(args, p, g2: dict) -> None:
    """MF-A: manifest PASS + sha match + direct mining-row/eval-id intersection == 0
    for EVERY family, BEFORE any W1 dispatch (plan §4 P1.8 / §7)."""
    t = _t2552()
    out_path = p.agg / "mfa_disjointness.json"
    regime = json.loads(REGIME_JSON.read_text())
    manifest = json.loads(
        (p.inputs / args.hf_prefix / "analysis_tensors/mining_manifest.json").read_text()
    )
    fams = manifest.get("families", {})
    assert fams, "mining_manifest.json carries no families"
    eval_set = set(int(x) for x in g2["eval_ids"])
    report: dict[str, dict] = {}
    for fam, doc in fams.items():
        assert doc.get("eval_disjoint_assert") == "PASS", (
            f"mining manifest family {fam}: eval_disjoint_assert != PASS"
        )
        assert doc.get("eval_ids_sha256") == regime["eval_ids_sha256"], (
            f"mining manifest family {fam}: eval_ids sha mismatch vs committed regime.json"
        )
        report[fam] = {"manifest_assert": "PASS", "n_mining_rows": doc.get("n_mining_rows")}
    # manifest-carried mining row ids (P1 writer; #2552 r2 g3-M1) — the check is
    # DOUBLE-ENDED: the manifest-DECLARED pool ids AND the jsonl-REALIZED example
    # rows must each be eval-disjoint, and the realized rows must be a SUBSET of
    # the declared pool (the pool is the candidate set; top-25 selection realizes
    # a subset of it)
    ids_npz = np.load(p.inputs / args.hf_prefix / "analysis_tensors/mining_row_ids.npz")
    for fam in ALL_FAMILIES:
        assert fam in ids_npz.files, (
            f"mining_row_ids.npz missing family {fam} — have {sorted(ids_npz.files)}"
        )
        manifest_ids = {int(x) for x in np.asarray(ids_npz[fam], np.int64)}
        rows = {int(r["row_id"]) for r in _mining_records(p, args.hf_prefix, fam)}
        stray = rows - manifest_ids
        assert not stray, (
            f"MF-A: family {fam} jsonl-realized mining rows outside the manifest-declared "
            f"pool (n={len(stray)}, e.g. {sorted(stray)[:5]})"
        )
        overlap = (rows | manifest_ids) & eval_set
        assert not overlap, (
            f"MF-A VIOLATION: family {fam} mining rows intersect realized eval ids "
            f"(n={len(overlap)})"
        )
        report.setdefault(fam, {})["direct_check_n_overlap"] = 0
        report[fam]["n_mining_row_ids"] = len(rows)
        report[fam]["n_manifest_pool_ids"] = len(manifest_ids)
        report[fam]["realized_subset_of_declared"] = True
    t.C.write_json_atomic(
        out_path,
        {
            "families": report,
            "eval_ids_sha256": g2["eval_ids_sha256"],
            "n_eval_realized": g2["n_eval_realized"],
            **as_metadata_dict(git_provenance(), phase="judge-mfa"),
        },
    )
    logger.info("[mfa] disjointness PASS for %d families -> %s", len(ALL_FAMILIES), out_path)


# ── eval-turn text fetch (checkpointed chunk sweep) ──────────────────────────────


def _texts_path(p) -> Path:
    return p.work / "eval_texts.jsonl"


def _fetch_eval_texts(args, p, eval_ids: np.ndarray, row_ci: np.ndarray) -> None:
    """Stream the rollout chunks (pinned revision, persistent chunk index) and persist
    each realized eval turn's response text (capped) — checkpointed per row, resumable.
    Text is never logged (digest-only)."""
    t = _t2552()
    out = _texts_path(p)
    state_path = p.work / "eval_texts_state.json"
    fingerprint = {"eval_ids_sha256": _sha_ids(eval_ids), "text_cap": EVAL_TURN_TEXT_CAP}
    if state_path.exists():
        prev = json.loads(state_path.read_text())
        assert prev == fingerprint, (
            f"eval_texts_state.json regime drift: {prev} != {fingerprint} — move the stale "
            "eval_texts aside rather than silently reusing another regime's texts"
        )
    else:
        t.C.write_json_atomic(state_path, fingerprint)
    have: set[int] = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    have.add(int(json.loads(line)["row_id"]))
    todo = [int(r) for r in eval_ids if int(r) not in have]
    if not todo:
        logger.info("[texts] resume: all %d eval turns present", len(eval_ids))
        return
    needed_ci = {int(row_ci[r]): int(r) for r in todo}
    assert len(needed_ci) == len(todo) and all(c >= 0 for c in needed_ci), (
        "eval ids must be text-resolvable (ci>=0; P0 guarantees this)"
    )
    ns = SimpleNamespace(out_root=p.work, max_chunks=args.max_chunks)
    n0 = len(have)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for k, (row_idx, ci, _prompt, response) in enumerate(
            t._iter_rows_pinned(ns, needed_ci, tag="eval_texts")
        ):
            rec = {
                "row_id": int(row_idx),
                "ci": int(ci),
                "text": response[:EVAL_TURN_TEXT_CAP],
                "n_chars_orig": len(response),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (k + 1) % 100 == 0:
                fh.flush()
                print(
                    f"[texts] unit {n0 + k + 1}/{len(eval_ids)} row={row_idx} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    with out.open(encoding="utf-8") as fh:
        n_final = sum(1 for line in fh if line.strip())
    assert n_final >= len(eval_ids), (
        f"[texts] recovered {n_final}/{len(eval_ids)} eval-turn texts (A11 join violated)"
    )
    logger.info("[texts] complete: %d turns at cap %d chars", n_final, EVAL_TURN_TEXT_CAP)


def _load_texts(p, eval_ids: np.ndarray) -> dict[int, str]:
    out: dict[int, str] = {}
    with _texts_path(p).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[int(rec["row_id"])] = rec["text"]
    missing = [int(r) for r in eval_ids if int(r) not in out]
    assert not missing, f"eval texts missing for {len(missing)} rows — rerun --wave prep"
    return out


def _require_g2(args, p) -> dict:
    path = p.agg / "g2_decision.json"
    assert path.exists(), "g2_decision.json missing — run --wave prep first"
    return json.loads(path.read_text())


# ── wave-specific parsers (rule-27 round-trip-tested) ────────────────────────────


def parse_w1(payload: object) -> str | None:
    if isinstance(payload, dict):
        d = payload.get("description")
        if isinstance(d, str) and d.strip():
            return d.strip()
    return None


def parse_w2(payload: object) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return None
    kept: dict[str, str] = {}
    lowered = {str(k).strip().lower(): v for k, v in payload.items()}
    for f in APP_D_FIELDS:
        v = lowered.get(f)
        if isinstance(v, str | int | float) and str(v).strip():
            kept[f] = str(v).strip()
    return kept if len(kept) >= W2_MIN_FIELDS else None


def parse_w3(payload: object) -> tuple[str, str] | None:
    """Return (field, category) with the category DERIVED from the schema (the
    judge's own category is recorded upstream but never trusted); ('none','none')
    is the explicit no-fit verdict."""
    if not isinstance(payload, dict):
        return None
    f = payload.get("field")
    if not isinstance(f, str):
        return None
    f = f.strip().lower()
    if f == "none":
        return ("none", "none")
    if f in FIELD_TO_CATEGORY:
        return (f, FIELD_TO_CATEGORY[f])
    return None


def parse_w4(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    c = payload.get("choice")
    if isinstance(c, str):
        c = c.strip().strip("()").strip().upper()
        if c in W4_LABELS:
            return c
    return None


def parse_w5(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    c = payload.get("choice")
    if isinstance(c, bool):
        return None
    if isinstance(c, int) and c in (1, 2):
        return c
    if isinstance(c, str):
        s = c.strip().lower().removeprefix("list").strip()
        if s in ("1", "2"):
            return int(s)
    return None


def parse_w6(payload: object) -> list[str] | None:
    r = payload.get("ranking") if isinstance(payload, dict) else payload
    if not isinstance(r, list) or len(r) != len(W6_LABELS):
        return None
    labels = [str(x).strip().upper() for x in r]
    return labels if sorted(labels) == sorted(W6_LABELS) else None


WAVE_PARSERS = {
    "w1": parse_w1,
    "w2": parse_w2,
    "w3": parse_w3,
    "w4": parse_w4,
    "w5": parse_w5,
    "w6": parse_w6,
}


# ── draw classification + reduce ────────────────────────────────────────────────


def classify_draw(d: object, wave_parser) -> tuple[str, object]:
    """Classify one stored draw dict -> (class, parsed_value_or_None).

    Classes: 'transport' | 'api_refusal' | 'truncation' | 'parse_fail' | 'valid'.
    Precedence mirrors the drop-class taxonomy (rules 9/23/24/28)."""
    if isinstance(d, dict) and d.get("error"):
        if is_transport_error_dict(d):
            return ("transport", None)
        if is_api_refusal_error_dict(d):
            return ("api_refusal", None)
        if is_truncation_error_dict(d):
            return ("truncation", None)
        raw = d.get("_raw_text")
        if isinstance(raw, str):
            v = wave_parser(parse_judge_json(raw))
            if v is not None:
                return ("valid", v)
        return ("parse_fail", None)
    if isinstance(d, dict):
        sr = d.get("stop_reason")
        if sr == "max_tokens":
            return ("truncation", None)
        if is_api_refusal_stop_reason(sr):
            return ("api_refusal", None)
        payload = {k: v for k, v in d.items() if k not in ("_raw_text", "stop_reason")}
        v = wave_parser(payload)
        if v is None and isinstance(d.get("_raw_text"), str):
            v = wave_parser(parse_judge_json(d["_raw_text"]))
        return ("valid", v) if v is not None else ("parse_fail", None)
    v = wave_parser(d)
    return ("valid", v) if v is not None else ("parse_fail", None)


def reduce_all_scores(all_scores: dict[str, object], wave_parser) -> dict[str, dict]:
    """Per-item reduce over the save_raw ``all_scores`` map (custom_id =
    '{item_id}__{idx:05d}__{comp_idx:02d}'). n_draws=1 => one draw per item."""
    per_item: dict[str, dict] = {}
    for custom_id, d in all_scores.items():
        item_id = custom_id.split("__")[0]
        cls, val = classify_draw(d, wave_parser)
        cur = per_item.get(item_id)
        # keep the best-class draw per item (a sync re-issue merge overlays later)
        rank = {"valid": 0, "parse_fail": 1, "truncation": 2, "api_refusal": 3, "transport": 4}
        if cur is None or rank[cls] < rank[cur["class"]]:
            per_item[item_id] = {"class": cls, "value": val}
    return per_item


def _arm_stats(item_ids: list[str], per_item: dict[str, dict]) -> dict:
    tally = {"valid": 0, "parse_fail": 0, "truncation": 0, "api_refusal": 0, "transport": 0}
    missing = 0
    for i in item_ids:
        rec = per_item.get(i)
        if rec is None:
            missing += 1
            continue
        tally[rec["class"]] += 1
    n = len(item_ids)
    n_valid = tally["valid"]
    return {
        "n_items": n,
        "n_missing_results": missing,
        **{f"n_{k}": v for k, v in tally.items()},
        "frac_items_complete": (n_valid / n) if n else float("nan"),
        "floor": FRAC_ITEMS_FLOOR,
        "below_floor": bool(n and (n_valid / n) < FRAC_ITEMS_FLOOR),
    }


def _reload_per_item(p, wave: str, base: str) -> dict[str, dict]:
    """Rebuild a completed wave's FINAL per-item reduce from its persisted raw
    (base batch file + the rule-28 sync re-issue overlay) with ZERO API calls —
    the resume path when ``run_wave`` skips a done wave and returns None. The
    overlay merge mirrors run_wave's in-band merge exactly (#2552 r2 g5-M1)."""
    parser = WAVE_PARSERS[base]
    raw = p.work / "raw" / wave / f"judge_raw_{wave}.json"
    # message covers BOTH callers (r2 g1 Minor 3): the resume path (meta says done,
    # raw lost) and the phase_w7 PRIMARY read (parent wave simply never ran)
    assert raw.exists(), (
        f"[{wave}] persisted raw missing: {raw} — either the wave never ran "
        f"(run --wave {wave} first) or its raw was lost after judge_meta was written"
    )
    per = reduce_all_scores(_load_all_scores(raw), parser)
    reissue = raw.with_name(f"judge_raw_{wave}_syncreissue.json")
    if reissue.exists():
        for i, rec in reduce_all_scores(_load_all_scores(reissue), parser).items():
            if rec["class"] == "valid" or per.get(i, {}).get("class") != "valid":
                per[i] = {**rec, "via": "sync_reissue"}
    return per


# ── dispatch + raw persistence ───────────────────────────────────────────────────


def _assert_item_ids(items: list[tuple[str, str]]) -> None:
    for item_id, _q in items:
        assert _ITEM_ID_RE.match(item_id) and "__" not in item_id, (
            f"item_id violates the batch custom-id grammar: {item_id!r}"
        )


def _dispatch(
    args,
    *,
    wave: str,
    items: list[tuple[str, str]],
    system: str,
    max_tokens: int,
    cache_dir: Path,
    save_raw: Path,
    judge_model: str,
    force_sync: bool = False,
    dry_run: bool = False,
) -> None:
    """One judge_completions_batch call over (item_id, question_block) items.

    n_draws=1: completions = {item_id: {block: [""]}}; the empty completion is
    unused by _user_msg. NO assistant prefill exists anywhere on this path."""
    _assert_item_ids(items)
    completions = {item_id: {block: [""]} for item_id, block in items}
    with keep_raw_judge_text():
        judge_completions_batch(
            completions,
            judge_system_prompt=system,
            format_user_msg=_user_msg,
            judge_model=judge_model,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            save_raw=save_raw,
            threshold_base=WAVE_THRESHOLD_BASE,
            force_sync=force_sync,
            dry_run=dry_run,
            checkpoint_dir=cache_dir / ".dispatch" / wave,
        )


def _load_all_scores(save_raw: Path) -> dict[str, object]:
    doc = json.loads(save_raw.read_text())
    return doc.get("all_scores", {})


def _upload_raw(args, p, wave: str, stage_files: list[Path]) -> None:
    """Persist raw judge outputs (full JSON incl. rationales) to HF
    ``<hf_prefix>/raw_completions/judge/<wave>/`` BEFORE any reduction consumes them.
    Text >9.5 MB line-splits into <9 MB shards (never gzip)."""
    if args.skip_upload or args.smoke or args.dry_run:
        logger.warning("[%s] raw upload SKIPPED (skip_upload/smoke/dry_run) — loud", wave)
        return
    import shutil

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    repo = _t2552().C.HF_DATA_REPO
    stage = p.work / "raw_stage" / wave
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for f in stage_files:
        shutil.copy2(f, stage / f.name)
    prefix = f"{args.hf_prefix}/raw_completions/judge/{wave}"
    res = upload_dir_sharded(
        stage,
        repo,
        prefix,
        repo_type="dataset",
        shard_glob="*",
        verify=True,
        delete_local=False,
        resume_skip=False,
    )
    if not res.rerouted:
        expected = [f"{prefix}/{q.name}" for q in sorted(stage.glob("*")) if q.is_file()]
        missing = hub.verify_repo_paths_uploaded(HfApi(), repo, expected, path_in_repo=prefix)
        assert not missing, f"[{wave}] raw upload verify FAILED — missing: {missing}"
    logger.info("[%s] raw persisted -> %s (%d files)", wave, prefix, len(stage_files))


def _stage_draws_jsonl(p, wave: str, tag: str, all_scores: dict[str, object]) -> list[Path]:
    """Serialize per-draw records (raw text + stop_reason + parse payloads) to
    sharded JSONL under the work root; returns the shard paths."""
    t = _t2552()
    rows = [{"custom_id": cid, "result": d} for cid, d in sorted(all_scores.items())]
    base = p.work / "raw" / wave / f"judge_draws_{wave}_{tag}.jsonl"
    return t._jsonl_write_sharded(base, rows)


# ── the generic wave runner ──────────────────────────────────────────────────────


def _wave_regime_key(args, g2: dict, wave: str, system: str, judge_model: str) -> dict:
    return {
        "wave": wave,
        "eval_ids_sha256": g2["eval_ids_sha256"],
        "rep_panel_sha256": g2["rep_panel_sha256"],
        "judge_model": judge_model,
        "max_tokens": MAX_TOKENS.get(wave, W7_MAX_TOKENS),
        "system_sha256": _sha_text(system),
        "seed": SEED,
        "threshold_base": WAVE_THRESHOLD_BASE,
        "turn_text_cap": EVAL_TURN_TEXT_CAP,
        "w4_candidate_cap": W4_CANDIDATE_CAP,
    }


def _wave_done(p, wave: str, regime_key: dict) -> bool:
    meta = p.agg / f"judge_meta_{wave}.json"
    if not meta.exists():
        return False
    doc = json.loads(meta.read_text())
    if doc.get("regime_key") == regime_key:
        # rule-29 floor binding on resume (#2552 r3 judge-completeness-floor): a done
        # marker written by PRE-floor-gate code (no below_floor_arms key), or one
        # recording below-floor arms with NO recorded waiver, must not short-circuit
        # the gate into a below-floor aggregate rebuild — quarantine + re-run instead.
        # No smoke conditional here: a below-floor smoke wave records waiver kind
        # "smoke" at write time (run_wave meta), so its resume passes on the waiver.
        assert "below_floor_arms" in doc, (
            f"judge_meta_{wave}.json predates the rule-29 completeness-floor gate "
            "(no below_floor_arms key) — quarantine it and re-run the wave"
        )
        assert not doc["below_floor_arms"] or doc.get("below_floor_waiver"), (
            f"judge_meta_{wave}.json records below-floor arms {doc['below_floor_arms']} "
            "with NO recorded waiver — quarantine it and re-run (rule 29)"
        )
        logger.info("[%s] resume: judge_meta present + regime match; skip", wave)
        return True
    raise AssertionError(
        f"judge_meta_{wave}.json exists with a DIFFERENT regime key — quarantine it before "
        "re-running (never silently overwrite another regime's wave)"
    )


def _require_pilot_pass(p, label: str) -> None:
    """Refuse a production dispatch whose MF-D pilot has not PASSed (label-keyed)."""
    path = p.agg / f"pilot_report_{label}.json"
    if not path.exists():
        print(f"[{label}] PILOT MISSING — run the pilot phase first", flush=True)
        raise SystemExit(RC_PILOT_FAIL)
    doc = json.loads(path.read_text())
    if doc.get("verdict") != "PASS":
        print(
            f"[{label}] PILOT verdict={doc.get('verdict')} — production dispatch refused",
            flush=True,
        )
        raise SystemExit(RC_PILOT_FAIL)


def run_wave(
    args,
    p,
    g2: dict,
    *,
    wave: str,
    arms: dict[str, list[tuple[str, str]]],
    system: str,
    judge_model: str = JUDGE_MODEL,
    base_wave: str | None = None,
    pilot_label: str | None = None,
) -> dict[str, dict] | None:
    """Dispatch -> persist raw -> reduce -> rule-28 sync re-issue -> merge -> meta.

    ``wave`` is the dispatch-group id (cache/meta/raw namespace: ``w1``, ``w1pt``,
    ``w7w3``, ``smoke_w2``); ``base_wave`` names which INSTRUMENT family it runs
    (parser + max_tokens lookup; defaults to ``wave``). ``pilot_label`` names the
    MF-D pilot report gating this dispatch (None => not pilot-gated: sub-5k waves
    w2/w6, W7 calibration, smoke probes). Returns the per-item reduce map (None
    when dry-run / already done)."""
    base = base_wave or wave
    max_tokens = MAX_TOKENS.get(base, W7_MAX_TOKENS)
    parser = WAVE_PARSERS.get(base)
    if parser is None:
        raise KeyError(f"no parser registered for wave {wave} (base {base})")
    regime_key = _wave_regime_key(args, g2, wave, system, judge_model)
    regime_key["max_tokens"] = max_tokens
    if not args.dry_run and _wave_done(p, wave, regime_key):
        return None
    items = [it for arm_items in arms.values() for it in arm_items]
    n_calls = len(items)
    logger.info("[%s] composed %d items across %d arms", wave, n_calls, len(arms))
    cache_dir = p.work / "judge_cache" / wave
    save_raw = p.work / "raw" / wave / f"judge_raw_{wave}.json"
    if args.dry_run:
        _dispatch(
            args,
            wave=wave,
            items=items,
            system=system,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            save_raw=save_raw,
            judge_model=judge_model,
            dry_run=True,
        )
        t = _t2552()
        t.C.write_json_atomic(
            p.work / f"dryrun_{wave}.json",
            {
                "wave": wave,
                "n_items": n_calls,
                "arms": {a: len(v) for a, v in arms.items()},
                "system_sha256": _sha_text(system),
                "max_tokens": max_tokens,
                "judge_model": judge_model,
            },
        )
        logger.info("[%s] dry-run: %d items, zero API calls", wave, n_calls)
        return None
    if pilot_label is not None and not args.smoke:
        _require_pilot_pass(p, pilot_label)
    t0 = time.time()
    _dispatch(
        args,
        wave=wave,
        items=items,
        system=system,
        max_tokens=max_tokens,
        cache_dir=cache_dir,
        save_raw=save_raw,
        judge_model=judge_model,
        force_sync=bool(args.smoke),
    )
    all_scores = _load_all_scores(save_raw)
    shards = _stage_draws_jsonl(p, wave, "batch", all_scores)
    _upload_raw(args, p, wave, [save_raw, *shards])  # BEFORE reduction (plan §10)

    per_item = reduce_all_scores(all_scores, parser)
    # expected-but-missing items carry NO persisted result — transport-lost by
    # definition (rule 24), so they join the re-issue set (#2552 r2 codex
    # judge-completeness-floor)
    missing = sorted({i for i, _q in items} - set(per_item))
    if missing:
        logger.warning(
            "[%s] %d expected items MISSING from persisted results — treated as "
            "transport-lost and re-issued (rule 24)",
            wave,
            len(missing),
        )
    censored = [
        i for i, rec in per_item.items() if rec["class"] in ("api_refusal", "transport")
    ] + missing
    n_sync = 0
    if censored and not args.smoke:
        logger.warning(
            "[%s] rule-28 sync re-issue: %d censored items (identical instrument)",
            wave,
            len(censored),
        )
        reissue_items = [(i, q) for i, q in items if i in set(censored)]
        save_raw2 = p.work / "raw" / wave / f"judge_raw_{wave}_syncreissue.json"
        _dispatch(
            args,
            wave=wave,
            items=reissue_items,
            system=system,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            save_raw=save_raw2,
            judge_model=judge_model,
            force_sync=True,
        )
        scores2 = _load_all_scores(save_raw2)
        shards2 = _stage_draws_jsonl(p, wave, "syncreissue", scores2)
        _upload_raw(args, p, f"{wave}", [save_raw2, *shards2])
        per_item2 = reduce_all_scores(scores2, parser)
        for i, rec in per_item2.items():
            if rec["class"] == "valid" or per_item.get(i, {}).get("class") != "valid":
                if rec["class"] == "valid":
                    n_sync += 1
                per_item[i] = {**rec, "via": "sync_reissue"}
    per_arm = {
        a: _arm_stats([it[0] for it in arm_items], per_item) for a, arm_items in arms.items()
    }
    below = [a for a, s in per_arm.items() if s["below_floor"]]
    if below:
        logger.warning(
            "[%s] BELOW frac_items_complete floor after re-issue: %s — drop-class triage "
            "required before any DV is plotted (rule 29)",
            wave,
            below,
        )
        if not (args.smoke or args.allow_below_floor):
            # HARD GATE (#2552 r2 codex judge-completeness-floor): aggregation HALTS
            # while any arm sits below the registered 0.95 floor. judge_meta is NOT
            # written, so a re-run resumes from the persisted raw + judge cache;
            # --allow-below-floor is the recorded triage-complete waiver.
            t = _t2552()
            t.C.write_json_atomic(
                p.agg / f"floor_fail_{wave}.json",
                {
                    "wave": wave,
                    "floor": FRAC_ITEMS_FLOOR,
                    "below_floor_arms": below,
                    "arms": per_arm,
                    "n_censored_reissued": len(censored),
                    "n_valid_from_sync_reissue": n_sync,
                    **as_metadata_dict(git_provenance(), phase=f"judge-{wave}-floorfail"),
                },
            )
            print(
                f"[{wave}] HALT: arms below the {FRAC_ITEMS_FLOOR} frac_items_complete "
                f"floor: {below} — triage the drop classes (rule 29; report at "
                f"floor_fail_{wave}.json); re-run resumes from persisted raw; "
                "--allow-below-floor records a waiver",
                flush=True,
            )
            raise SystemExit(RC_FLOOR_FAIL)
    meta = {
        "wave": wave,
        "regime_key": regime_key,
        "judge_model": judge_model,
        "max_tokens": max_tokens,
        "temperature": "API default (not threaded; judge_dispatch contract)",
        "transport": "batch (threshold_base=0)" if not args.smoke else "sync (smoke)",
        "n_items": n_calls,
        "n_draws": 1,
        "arms": per_arm,
        "below_floor_arms": below,
        # a smoke run that proceeds below-floor records an explicit waiver of kind
        # "smoke" (r3): _wave_done can then bind the floor on resume with NO
        # smoke-conditional of its own — the exemption is durably in the artifact.
        "below_floor_waiver": bool(below and (args.allow_below_floor or args.smoke)),
        "below_floor_waiver_kind": (
            ("flag" if args.allow_below_floor else "smoke")
            if below and (args.allow_below_floor or args.smoke)
            else None
        ),
        "batch_sync_split": {
            "n_valid_from_sync_reissue": n_sync,
            "n_censored_reissued": len(censored),
        },
        "n_eval_realized": g2["n_eval_realized"],
        "descoped": g2["descoped"],
        "wall_s": round(time.time() - t0, 1),
        **as_metadata_dict(git_provenance(), phase=f"judge-{wave}"),
    }
    t = _t2552()
    # a floor_fail report from an earlier HALTed attempt is stale the moment the wave
    # passes (or is waived) — unlink so a triage reader never sees a floor-fail beside
    # a passing meta (#2552 r2 g1 Minor 1)
    (p.agg / f"floor_fail_{wave}.json").unlink(missing_ok=True)
    t.C.write_json_atomic(p.agg / f"judge_meta_{wave}.json", meta)
    logger.info(
        "[%s] done: %d items, sync_reissue_valid=%d, below_floor=%s, wall=%.0fs",
        wave,
        n_calls,
        n_sync,
        below,
        time.time() - t0,
    )
    return per_item


# ── item composition per wave ────────────────────────────────────────────────────


def _grouped_mining(p, hf_prefix: str, family: str) -> dict[int, list[dict]]:
    by_feat: dict[int, list[dict]] = {}
    for rec in _mining_records(p, hf_prefix, family):
        by_feat.setdefault(int(rec["feat_id"]), []).append(rec)
    for recs in by_feat.values():
        recs.sort(key=lambda r: int(r["rank"]))
    return by_feat


def _w1_block_ta(recs: list[dict]) -> str:
    parts = ["FEATURE EXAMPLES (top-activating turns, activation shown):", ""]
    for k, r in enumerate(recs):
        parts.append(f"### Example {k + 1} (activation={float(r['activation']):.4f})")
        parts.append(str(r["text"]))
        parts.append("")
    return "\n".join(parts)


def _w1_block_pt(recs: list[dict]) -> str:
    parts = ["FEATURE EXAMPLES (top-activating token contexts):", ""]
    for k, r in enumerate(recs):
        peak_rel = int(r["peak_token_abs"]) - int(r.get("window_lo_abs", 0))
        # render offset AND activation VALUE per retained token (#2552 r2 codex
        # pt-activation-values) — offsets alone discard the activation profile
        pairs = [(int(t), float(a)) for t, a in r.get("window_token_acts", [])][:12]
        shown = ", ".join(f"{t}:{a:.3f}" for t, a in pairs)
        parts.append(
            f"### Example {k + 1} (peak activation={float(r['activation']):.4f}; "
            f"peak token at offset {peak_rel} of this window; "
            f"activating token offsets with activations: [{shown}])"
        )
        parts.append(str(r["window_text"]))
        parts.append("")
    return "\n".join(parts)


def compose_w1(args, p, g2: dict) -> dict[str, list[tuple[str, str]]]:
    """W1 arms keyed by description family; item per feature in the family's
    realized description-need set; examples from the mining jsonls."""
    lists_doc = _load_lists(p, args.hf_prefix)
    mats = _mat_panels()
    need = compute_need_sets(
        set(int(x) for x in g2["eval_ids"]),
        lists_doc,
        np.asarray(g2["rep_panel_ids"], np.int64),
        mats,
    )
    arms: dict[str, list[tuple[str, str]]] = {}
    for fam in ALL_FAMILIES:
        mined = _grouped_mining(p, args.hf_prefix, fam)
        missing = sorted(f for f in need[fam] if f not in mined)
        assert not missing, (
            f"[w1] family {fam}: {len(missing)} needed features absent from mining "
            f"(first: {missing[:5]}) — mining must cover the description-need set"
        )
        block = _w1_block_pt if fam == "pt" else _w1_block_ta
        arms[fam] = [(f"w1-{fam}-f{feat}", block(mined[feat])) for feat in sorted(need[fam])]
    return arms


def compose_w2(args, p, g2: dict) -> dict[str, list[tuple[str, str]]]:
    eval_ids = np.asarray(g2["eval_ids"], np.int64)
    texts = _load_texts(p, eval_ids)
    items = [
        (f"w2-r{int(r)}", f"CONVERSATION TURN (assistant response):\n\n{texts[int(r)]}")
        for r in eval_ids
    ]
    return {"all_turns": items}


def _descriptions(p, family: str) -> dict[int, str]:
    path = p.agg / f"descriptions_{family}.json"
    assert path.exists(), f"descriptions_{family}.json missing — run --wave w1 first"
    doc = json.loads(path.read_text())
    return {int(k): v for k, v in doc["descriptions"].items()}


def compose_w3(args, p, g2: dict) -> dict[str, list[tuple[str, str]]]:
    """W3 arms keyed by TA dictionary; item per PANEL feature (rep: realized rep
    panel; mat_*: banked union-npz feat ids)."""
    mats = _mat_panels()
    panels = {
        "rep_ta": np.asarray(g2["rep_panel_ids"], np.int64),
        "mat_k100": mats["mat_k100"],
        "mat_k200": mats["mat_k200"],
    }
    arms: dict[str, list[tuple[str, str]]] = {}
    for fam, panel in panels.items():
        desc = _descriptions(p, fam)
        mined = _grouped_mining(p, args.hf_prefix, fam)
        items: list[tuple[str, str]] = []
        n_missing_desc = 0
        for feat in sorted(int(x) for x in panel):
            if feat not in desc:
                n_missing_desc += 1
                continue
            excerpts = [str(r["text"]) for r in mined.get(feat, [])[:W3_EXCERPTS]]
            block_parts = [f"FEATURE DESCRIPTION:\n{desc[feat]}", ""]
            block_parts.append("TOP EXAMPLE EXCERPTS:")
            for k, e in enumerate(excerpts):
                block_parts.append(f"### Excerpt {k + 1}\n{e}\n")
            items.append((f"w3-{fam}-f{feat}", "\n".join(block_parts)))
        if n_missing_desc:
            logger.warning(
                "[w3] %s: %d panel features lack a valid W1 description (excluded, recorded)",
                fam,
                n_missing_desc,
            )
        arms[fam] = items
    return arms


def w4_presentation(row_id: int, pool: list[int]) -> dict:
    """Deterministic per-turn candidate set + order (seed 2552; FIXED across
    configs — the rng key carries only the row id)."""
    others = sorted(int(x) for x in pool if int(x) != int(row_id))
    rng = np.random.default_rng([SEED, 4, int(row_id)])
    distractors = [int(x) for x in rng.choice(others, size=9, replace=False)]
    cands = [int(row_id), *distractors]
    order = [cands[i] for i in rng.permutation(10)]
    gold_label = W4_LABELS[order.index(int(row_id))]
    return {"candidates": order, "gold_label": gold_label}


def _desc_list_lines(feats: list[int], desc: dict[int, str]) -> tuple[list[str], int]:
    lines: list[str] = []
    n_missing = 0
    for f in feats:
        d = desc.get(int(f))
        if d is None:
            n_missing += 1
            continue
        lines.append(f"- {d}")
    return lines, n_missing


def compose_w4(args, p, g2: dict) -> tuple[dict[str, list[tuple[str, str]]], dict[str, dict], dict]:
    """W4 arms keyed by config. Returns (arms, row_meta by item_id, presentation)."""
    eval_ids = [int(x) for x in g2["eval_ids"]]
    texts = _load_texts(p, np.asarray(eval_ids, np.int64))
    lists_doc = _load_lists(p, args.hf_prefix)
    desc_by_fam = {fam: _descriptions(p, fam) for fam in ALL_FAMILIES}
    pres = {int(r): w4_presentation(int(r), eval_ids) for r in eval_ids}
    arms: dict[str, list[tuple[str, str]]] = {}
    row_meta: dict[str, dict] = {}
    for cfg in CONFIGS:
        turn_feats = _turn_lists(lists_doc, cfg, set(eval_ids))
        desc = desc_by_fam[CONFIG_FAMILY[cfg]]
        items: list[tuple[str, str]] = []
        for r in eval_ids:
            feats = turn_feats.get(r, [])
            lines, n_missing = _desc_list_lines(feats, desc)
            item_id = f"w4-{cfg}-r{r}"
            if not lines:
                row_meta[item_id] = {
                    "row_id": r,
                    "config": cfg,
                    "valid": False,
                    "reason": "no_described_features",
                    "n_missing_desc": n_missing,
                }
                continue
            pp = pres[r]
            parts = ["FEATURE DESCRIPTIONS (derived from one of the candidate turns):", ""]
            parts.extend(lines)
            parts.append("")
            parts.append("CANDIDATE TURNS:")
            for lab, cand in zip(W4_LABELS, pp["candidates"], strict=True):
                parts.append(f"### {lab}\n{texts[cand][:W4_CANDIDATE_CAP]}\n")
            items.append((item_id, "\n".join(parts)))
            row_meta[item_id] = {
                "row_id": r,
                "config": cfg,
                "valid": None,
                "gold": pp["gold_label"],
                "n_desc": len(lines),
                "n_missing_desc": n_missing,
            }
        arms[cfg] = items
    return arms, row_meta, {str(k): v for k, v in pres.items()}


def _render_summary(fields: dict[str, str]) -> str:
    return "\n".join(f"{f}: {fields[f]}" for f in APP_D_FIELDS if f in fields)


def _load_summaries(p, g2: dict) -> dict[int, dict[str, str]]:
    path = p.agg / f"summaries_{g2['n_eval_realized']}.json"
    assert path.exists(), f"{path.name} missing — run --wave w2 first"
    doc = json.loads(path.read_text())
    return {int(k): v for k, v in doc["summaries"].items()}


def w5_assignment(row_id: int, pair_idx: int) -> bool:
    """True => the pair's FIRST config is presented as List 1 (seed 2552)."""
    rng = np.random.default_rng([SEED, 5, int(row_id), int(pair_idx)])
    return bool(rng.integers(0, 2) == 0)


def compose_w5(args, p, g2: dict) -> tuple[dict[str, list[tuple[str, str]]], dict[str, dict]]:
    """W5 arms keyed by 'cfgA__vs__cfgB'... (single-underscore-safe: 'cfgA-vs-cfgB')."""
    eval_ids = [int(x) for x in g2["eval_ids"]]
    summaries = _load_summaries(p, g2)
    lists_doc = _load_lists(p, args.hf_prefix)
    desc_by_fam = {fam: _descriptions(p, fam) for fam in ALL_FAMILIES}
    turn_feats = {cfg: _turn_lists(lists_doc, cfg, set(eval_ids)) for cfg in CONFIGS}
    arms: dict[str, list[tuple[str, str]]] = {}
    row_meta: dict[str, dict] = {}
    for pi, (a, b) in enumerate(W5_PAIRS):
        arm_key = f"{a}-vs-{b}"
        items: list[tuple[str, str]] = []
        for r in eval_ids:
            item_id = f"w5-r{r}-{pi}"
            summ = summaries.get(r)
            la, _ = _desc_list_lines(turn_feats[a].get(r, []), desc_by_fam[CONFIG_FAMILY[a]])
            lb, _ = _desc_list_lines(turn_feats[b].get(r, []), desc_by_fam[CONFIG_FAMILY[b]])
            if summ is None or not la or not lb:
                row_meta[item_id] = {
                    "row_id": r,
                    "pair": [a, b],
                    "valid": False,
                    "reason": "missing_summary_or_empty_list",
                }
                continue
            k = min(len(la), len(lb))  # equal-length truncation rule (must-ask to change)
            la, lb = la[:k], lb[:k]
            first_is_a = w5_assignment(r, pi)
            l1, l2 = (la, lb) if first_is_a else (lb, la)
            block = (
                "STRUCTURED SUMMARY OF THE TURN:\n"
                + _render_summary(summ)
                + "\n\nLIST 1:\n"
                + "\n".join(l1)
                + "\n\nLIST 2:\n"
                + "\n".join(l2)
            )
            items.append((item_id, block))
            row_meta[item_id] = {
                "row_id": r,
                "pair": [a, b],
                "valid": None,
                "list1": a if first_is_a else b,
                "list2": b if first_is_a else a,
                "list_len": k,
            }
        arms[arm_key] = items
    return arms, row_meta


def w6_assignment(row_id: int) -> dict[str, str]:
    rng = np.random.default_rng([SEED, 6, int(row_id)])
    order = [CONFIGS[i] for i in rng.permutation(len(CONFIGS))]
    return dict(zip(W6_LABELS, order, strict=True))


def compose_w6(args, p, g2: dict) -> tuple[dict[str, list[tuple[str, str]]], dict[str, dict]]:
    eval_ids = [int(x) for x in g2["eval_ids"]]
    summaries = _load_summaries(p, g2)
    lists_doc = _load_lists(p, args.hf_prefix)
    desc_by_fam = {fam: _descriptions(p, fam) for fam in ALL_FAMILIES}
    turn_feats = {cfg: _turn_lists(lists_doc, cfg, set(eval_ids)) for cfg in CONFIGS}
    items: list[tuple[str, str]] = []
    row_meta: dict[str, dict] = {}
    for r in eval_ids:
        item_id = f"w6-r{r}"
        summ = summaries.get(r)
        assign = w6_assignment(r)
        rendered: dict[str, list[str]] = {}
        ok = summ is not None
        for lab in W6_LABELS:
            cfg = assign[lab]
            lines, _m = _desc_list_lines(
                turn_feats[cfg].get(r, []), desc_by_fam[CONFIG_FAMILY[cfg]]
            )
            rendered[lab] = lines
            ok = ok and bool(lines)
        if not ok:
            row_meta[item_id] = {
                "row_id": r,
                "valid": False,
                "reason": "missing_summary_or_empty_list",
                "assignment": assign,
            }
            continue
        parts = ["STRUCTURED SUMMARY OF THE TURN:", _render_summary(summ), ""]
        for lab in W6_LABELS:
            parts.append(f"LIST {lab}:")
            parts.extend(rendered[lab])
            parts.append("")
        items.append((item_id, "\n".join(parts)))
        row_meta[item_id] = {"row_id": r, "valid": None, "assignment": assign}
    return {"all_turns": items}, row_meta


# ── MF-D pilots (rule-26 clauses via the production instrument) ──────────────────


def _pilot_sample(arms: dict[str, list[tuple[str, str]]]) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for k, (arm, items) in enumerate(sorted(arms.items())):
        assert len(items) >= PILOT_MIN_EFFECTIVE, (
            f"pilot arm {arm} holds {len(items)} items < the {PILOT_MIN_EFFECTIVE}-draw floor "
            "(#2124 satisfiability)"
        )
        rng = np.random.default_rng([SEED, 26, k])
        idx = rng.choice(len(items), size=PILOT_MIN_EFFECTIVE, replace=False)
        out[arm] = [items[int(i)] for i in sorted(idx)]
    return out


def run_pilot(
    args,
    p,
    g2: dict,
    *,
    wave: str,
    arms: dict[str, list[tuple[str, str]]],
    system: str,
    label: str | None = None,
) -> None:
    """MF-D pilot on the wave's OWN batch route with a FRESH cache; gates:
    (a) zero stop_reason=='max_tokens'; (b) per-arm parse-fail < 2% under the wave's
    own parser; (c) route parity (realized path=='batch', n_cached==0);
    (d) per-arm api-refusal < 0.10; effective draws >= 51/arm.

    ``label`` names the pilot report (defaults to ``wave``); a prior PASS report
    under the same label is honored (skip) — a FAIL report re-runs a fresh attempt."""
    t = _t2552()
    label = label or wave
    report_path = p.agg / f"pilot_report_{label}.json"
    if report_path.exists() and not args.dry_run:
        prior = json.loads(report_path.read_text())
        if prior.get("verdict") == "PASS":
            logger.info("[pilot-%s] resume: prior PASS report present; skip", label)
            return
        logger.warning("[pilot-%s] prior verdict=%s — fresh attempt", label, prior.get("verdict"))
    max_tokens = MAX_TOKENS[wave]
    parser = WAVE_PARSERS[wave]
    wave_n_calls = sum(len(v) for v in arms.values())
    pilot_arms = _pilot_sample(arms)
    attempt = 0
    while (p.work / "pilot" / label / f"attempt_{attempt}").exists():
        attempt += 1
    root = p.work / "pilot" / label / f"attempt_{attempt}"
    per_arm_report: dict[str, dict] = {}
    failures: list[str] = []
    for arm, items in pilot_arms.items():
        cache_dir = root / "cache" / arm
        save_raw = root / f"judge_raw_pilot_{arm}.json"
        if args.dry_run:
            _dispatch(
                args,
                wave=wave,
                items=items,
                system=system,
                max_tokens=max_tokens,
                cache_dir=cache_dir,
                save_raw=save_raw,
                judge_model=JUDGE_MODEL,
                dry_run=True,
            )
            continue
        _dispatch(
            args,
            wave=wave,
            items=items,
            system=system,
            max_tokens=max_tokens,
            cache_dir=cache_dir,
            save_raw=save_raw,
            judge_model=JUDGE_MODEL,
        )
        doc = json.loads(save_raw.read_text())
        routing = doc.get("routing") or {}
        if doc.get("n_cached", 0) != 0:
            failures.append(f"{arm}: n_cached={doc.get('n_cached')} != 0 (unverifiable)")
        if routing.get("path") != "batch" or routing.get("forced_sync"):
            failures.append(f"{arm}: realized route {routing.get('path')!r} != batch")
        per_item = reduce_all_scores(doc.get("all_scores", {}), parser)
        tally = {"valid": 0, "parse_fail": 0, "truncation": 0, "api_refusal": 0, "transport": 0}
        for rec in per_item.values():
            tally[rec["class"]] += 1
        n = len(items)
        n_effective = n - tally["transport"]
        n_answered = n - tally["transport"] - tally["api_refusal"]
        parse_rate = (tally["parse_fail"] / n_answered) if n_answered else 1.0
        refusal_rate = (tally["api_refusal"] / n_effective) if n_effective else 1.0
        if tally["truncation"]:
            failures.append(f"{arm}: {tally['truncation']} truncation draws (raise max_tokens)")
        if n_effective < PILOT_MIN_EFFECTIVE:
            failures.append(f"{arm}: effective draws {n_effective} < {PILOT_MIN_EFFECTIVE}")
        if parse_rate >= PILOT_PARSE_FAIL_THRESHOLD:
            failures.append(f"{arm}: parse-fail {parse_rate:.3f} >= {PILOT_PARSE_FAIL_THRESHOLD}")
        if refusal_rate >= PILOT_API_REFUSAL_THRESHOLD:
            failures.append(
                f"{arm}: api-refusal {refusal_rate:.3f} >= {PILOT_API_REFUSAL_THRESHOLD}"
            )
        per_arm_report[arm] = {
            **{f"n_{k}": v for k, v in tally.items()},
            "n_items": n,
            "parse_fail_rate": parse_rate,
            "api_refusal_rate": refusal_rate,
            "routing_path": routing.get("path"),
            "n_cached": doc.get("n_cached"),
        }
    if args.dry_run:
        logger.info(
            "[pilot-%s] dry-run: %d arms x %d items, zero API calls",
            wave,
            len(pilot_arms),
            PILOT_MIN_EFFECTIVE,
        )
        return
    verdict = "PASS" if not failures else "FAIL"
    report = {
        "wave": wave,
        "label": label,
        "verdict": verdict,
        "failures": failures,
        "arms": per_arm_report,
        "judge_model": JUDGE_MODEL,
        "max_tokens": max_tokens,
        "n_draws": 1,
        "target_draws_per_arm": PILOT_MIN_EFFECTIVE,
        "parse_fail_threshold": PILOT_PARSE_FAIL_THRESHOLD,
        "api_refusal_threshold": PILOT_API_REFUSAL_THRESHOLD,
        "wave_declaration": {
            "wave_n_calls": wave_n_calls,
            "wave_threshold_base": WAVE_THRESHOLD_BASE,
            "wave_force_sync": False,
        },
        "pilot_transport_note": (
            "pilot forced onto the wave's batch route via the SAME threshold_base=0 pin as "
            "production; fresh per-attempt cache (n_cached==0 asserted)"
        ),
        "reference": (
            "clauses mirror eval.judge_pilot.judge_pilot_gate (rule 26(a)-(d), #2124 sizing); "
            "re-implemented at the exact production instrument — " + PILOT_WAIVE_PARSE_FAIL_REASON
        ),
        "attempt_root": str(root),
        **as_metadata_dict(git_provenance(), phase=f"judge-pilot-{label}"),
    }
    t.C.write_json_atomic(report_path, report)
    logger.info("[pilot-%s] verdict=%s (%d arms)", label, verdict, len(pilot_arms))
    if verdict != "PASS":
        print(f"[pilot-{label}] FAIL: {failures}", flush=True)
        raise SystemExit(RC_PILOT_FAIL)


# ── wave phases (compose + run + aggregate) ──────────────────────────────────────


def _compose_w1_split(args, p, g2):
    """W1 splits into TA-system and PT-system dispatch groups sharing one wave id."""
    arms = compose_w1(args, p, g2)
    ta = {f: arms[f] for f in TA_FAMILIES}
    pt = {"pt": arms["pt"]}
    return ta, pt


def phase_pilot_w1(args) -> None:
    """Two W1 pilots: the 3 TA arms under the TA system prompt, and the pt arm
    under the pt system prompt (a different instrument => its own pilot report)."""
    p = _paths(args)
    g2 = _require_g2(args, p)
    ta, pt = _compose_w1_split(args, p, g2)
    run_pilot(args, p, g2, wave="w1", arms=ta, system=W1_SYSTEM_TA, label="w1")
    run_pilot(args, p, g2, wave="w1", arms=pt, system=W1_SYSTEM_PT, label="w1pt")


def phase_w1(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    ta, pt = _compose_w1_split(args, p, g2)
    per_ta = run_wave(args, p, g2, wave="w1", arms=ta, system=W1_SYSTEM_TA, pilot_label="w1")
    per_pt = run_wave(
        args,
        p,
        g2,
        wave="w1pt",
        base_wave="w1",
        arms=pt,
        system=W1_SYSTEM_PT,
        pilot_label="w1pt",
    )
    if args.dry_run:
        return
    t = _t2552()
    # resume-safe aggregate rebuild (#2552 r2 g5-M1): a done dispatch group returns
    # None — rebuild its per-item map from the persisted raw (+ sync overlay) and
    # ALWAYS rewrite the deterministic aggregates (kills the empty-aggregate shape)
    if per_ta is None:
        per_ta = _reload_per_item(p, "w1", "w1")
    if per_pt is None:
        per_pt = _reload_per_item(p, "w1pt", "w1")
    merged: dict[str, dict] = {**per_ta, **per_pt}
    for fam in ALL_FAMILIES:
        pref = f"w1-{fam}-f"
        descs = {
            int(i.removeprefix(pref)): rec["value"]
            for i, rec in merged.items()
            if i.startswith(pref) and rec["class"] == "valid"
        }
        n_drop = sum(
            1 for i, rec in merged.items() if i.startswith(pref) and rec["class"] != "valid"
        )
        t.C.write_json_atomic(
            p.agg / f"descriptions_{fam}.json",
            {
                "family": fam,
                "n_valid": len(descs),
                "n_dropped": n_drop,
                "descriptions": {str(k): v for k, v in sorted(descs.items())},
            },
        )
        logger.info("[w1] %s: %d descriptions (%d dropped)", fam, len(descs), n_drop)


def phase_w2(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    arms = compose_w2(args, p, g2)
    per_item = run_wave(args, p, g2, wave="w2", arms=arms, system=W2_SYSTEM)
    if args.dry_run:
        return
    if per_item is None:  # resumed: rebuild from persisted raw (#2552 r2 g5-M1)
        per_item = _reload_per_item(p, "w2", "w2")
    t = _t2552()
    summaries = {
        str(int(i.removeprefix("w2-r"))): rec["value"]
        for i, rec in per_item.items()
        if rec["class"] == "valid"
    }
    t.C.write_json_atomic(
        p.agg / f"summaries_{g2['n_eval_realized']}.json",
        {
            "n_valid": len(summaries),
            "n_items": sum(len(v) for v in arms.values()),
            "min_fields": W2_MIN_FIELDS,
            "summaries": summaries,
        },
    )
    logger.info("[w2] %d summaries", len(summaries))


def phase_pilot_w3(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    run_pilot(args, p, g2, wave="w3", arms=compose_w3(args, p, g2), system=W3_SYSTEM)


def phase_w3(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    arms = compose_w3(args, p, g2)
    per_item = run_wave(args, p, g2, wave="w3", arms=arms, system=W3_SYSTEM, pilot_label="w3")
    if args.dry_run:
        return
    if per_item is None:  # resumed: rebuild from persisted raw (#2552 r2 g5-M1)
        per_item = _reload_per_item(p, "w3", "w3")
    t = _t2552()
    for fam in TA_FAMILIES:
        pref = f"w3-{fam}-f"
        assigned: dict[str, dict] = {}
        n_none = 0
        n_drop = 0
        for i, rec in per_item.items():
            if not i.startswith(pref):
                continue
            if rec["class"] != "valid":
                n_drop += 1
                continue
            field, cat = rec["value"]
            if field == "none":
                n_none += 1
                continue
            assigned[i.removeprefix(pref)] = {"field": field, "category": cat}
        t.C.write_json_atomic(
            p.agg / f"w3_categories_{fam}.json",
            {
                "dictionary": fam,
                "n_assigned": len(assigned),
                "n_none": n_none,
                "n_dropped": n_drop,
                "assignments": assigned,
            },
        )
        logger.info(
            "[w3] %s: %d assigned / %d none / %d dropped", fam, len(assigned), n_none, n_drop
        )


def phase_pilot_w4(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    arms, _meta, _pres = compose_w4(args, p, g2)
    run_pilot(args, p, g2, wave="w4", arms=arms, system=W4_SYSTEM)


def phase_w4(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    arms, row_meta, pres = compose_w4(args, p, g2)
    t = _t2552()
    t.C.write_json_atomic(p.work / "w4_presentation.json", pres)
    per_item = run_wave(args, p, g2, wave="w4", arms=arms, system=W4_SYSTEM, pilot_label="w4")
    if args.dry_run:
        return
    if per_item is None:  # resumed: rebuild from persisted raw (#2552 r2 g5-M1)
        per_item = _reload_per_item(p, "w4", "w4")
    rows = []
    for item_id, meta in sorted(row_meta.items()):
        rec = per_item.get(item_id)
        row = dict(meta)
        row["item_id"] = item_id
        if meta.get("valid") is False:
            row["drop_class"] = row.pop("reason")
        elif rec is None or rec["class"] != "valid":
            row["valid"] = False
            row["drop_class"] = rec["class"] if rec else "missing_result"
        else:
            row["valid"] = True
            row["choice"] = rec["value"]
            row["correct"] = bool(rec["value"] == meta["gold"])
        rows.append(row)
    valid_by_cfg: dict[str, set[int]] = {c: set() for c in CONFIGS}
    correct_by_cfg: dict[str, int] = {c: 0 for c in CONFIGS}
    for row in rows:
        if row.get("valid"):
            valid_by_cfg[row["config"]].add(row["row_id"])
            correct_by_cfg[row["config"]] += int(row["correct"])
    inter = valid_by_cfg["rep_ta"] & valid_by_cfg["pt_max"]
    summary = {
        "per_config": {
            c: {
                "n_valid": len(valid_by_cfg[c]),
                "n_correct": correct_by_cfg[c],
                "accuracy": (correct_by_cfg[c] / len(valid_by_cfg[c]))
                if valid_by_cfg[c]
                else float("nan"),
            }
            for c in CONFIGS
        },
        "complete_pairs_rep_ta_pt_max": {
            "n": len(inter),
            "note": "H2 Δ_disc denominator: EXACT intersection of valid rep_ta and pt_max "
            "matching-verdict turn ids (plan §3 complete-pair rule)",
        },
        "n_eval_realized": g2["n_eval_realized"],
        "chance": 1.0 / len(W4_LABELS),
    }
    t.C.write_json_atomic(
        p.dere / "matching_perturn.json",
        {"rows": rows, "summary": summary, **as_metadata_dict(git_provenance(), phase="judge-w4")},
    )
    t.C.write_json_atomic(p.agg / "w4_matching_summary.json", summary)
    logger.info("[w4] rows=%d complete_pairs=%d", len(rows), len(inter))


def phase_pilot_w5(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    arms, _meta = compose_w5(args, p, g2)
    run_pilot(args, p, g2, wave="w5", arms=arms, system=W5_SYSTEM)


def phase_w5(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    arms, row_meta = compose_w5(args, p, g2)
    per_item = run_wave(args, p, g2, wave="w5", arms=arms, system=W5_SYSTEM, pilot_label="w5")
    if args.dry_run:
        return
    if per_item is None:  # resumed: rebuild from persisted raw (#2552 r2 g5-M1)
        per_item = _reload_per_item(p, "w5", "w5")
    t = _t2552()
    rows = []
    for item_id, meta in sorted(row_meta.items()):
        rec = per_item.get(item_id)
        row = dict(meta)
        row["item_id"] = item_id
        if meta.get("valid") is False:
            row["drop_class"] = row.pop("reason")
        elif rec is None or rec["class"] != "valid":
            row["valid"] = False
            row["drop_class"] = rec["class"] if rec else "missing_result"
        else:
            row["valid"] = True
            row["choice"] = rec["value"]
            row["winner"] = meta["list1"] if rec["value"] == 1 else meta["list2"]
        rows.append(row)
    wins: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("valid"):
            key = "-vs-".join(row["pair"])
            wins.setdefault(key, {"n_valid": 0})
            wins[key]["n_valid"] += 1
            wins[key][row["winner"]] = wins[key].get(row["winner"], 0) + 1
    summary = {"per_pair": wins, "n_eval_realized": g2["n_eval_realized"]}
    t.C.write_json_atomic(
        p.dere / "pairwise_perturn.json",
        {"rows": rows, "summary": summary, **as_metadata_dict(git_provenance(), phase="judge-w5")},
    )
    t.C.write_json_atomic(p.agg / "w5_pairwise_summary.json", summary)
    logger.info("[w5] rows=%d pairs=%d", len(rows), len(wins))


def phase_w6(args) -> None:
    p = _paths(args)
    g2 = _require_g2(args, p)
    arms, row_meta = compose_w6(args, p, g2)
    per_item = run_wave(args, p, g2, wave="w6", arms=arms, system=W6_SYSTEM)
    if args.dry_run:
        return
    if per_item is None:  # resumed: rebuild from persisted raw (#2552 r2 g5-M1)
        per_item = _reload_per_item(p, "w6", "w6")
    t = _t2552()
    rows = []
    rank_sums: dict[str, list[int]] = {c: [] for c in CONFIGS}
    for item_id, meta in sorted(row_meta.items()):
        rec = per_item.get(item_id)
        row = dict(meta)
        row["item_id"] = item_id
        if meta.get("valid") is False:
            row["drop_class"] = row.pop("reason")
        elif rec is None or rec["class"] != "valid":
            row["valid"] = False
            row["drop_class"] = rec["class"] if rec else "missing_result"
        else:
            row["valid"] = True
            ranking_cfgs = [meta["assignment"][lab] for lab in rec["value"]]
            row["ranking"] = ranking_cfgs
            for pos, cfg in enumerate(ranking_cfgs):
                rank_sums[cfg].append(pos + 1)
        rows.append(row)
    summary = {
        "mean_rank": {c: (float(np.mean(v)) if v else float("nan")) for c, v in rank_sums.items()},
        "n_valid": sum(1 for r in rows if r.get("valid")),
        "n_eval_realized": g2["n_eval_realized"],
    }
    t.C.write_json_atomic(
        p.agg / "w6_ranking_perturn.json",
        {"rows": rows, "summary": summary, **as_metadata_dict(git_provenance(), phase="judge-w6")},
    )
    logger.info("[w6] rows=%d n_valid=%d", len(rows), summary["n_valid"])


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (nan-nan at n=0)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p_ = k / n
    denom = 1 + z * z / n
    center = (p_ + z * z / (2 * n)) / denom
    half = z * float(np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n))) / denom
    return (float(center - half), float(center + half))


def _w7_cells(wave: str, item_id: str, primary_rec: dict) -> tuple[str, ...]:
    """Per-instrument calibration cell keys (#2552 r2 codex w7-calibration; r3):
    w3 -> the REGISTERED joint (family x primary-category) cell (plan line 95:
    "the 200 W3 items divide over 3 dictionaries x 5 categories ... report the
    realized per-cell n with a CI per cell") PLUS the family/category marginals
    as descriptive extras; w4 -> config; w5 -> config pair."""
    if wave == "w3":
        fam = item_id.split("-")[1]
        cat = primary_rec["value"][1]
        return (f"family={fam}|category={cat}", f"family={fam}", f"category={cat}")
    if wave == "w4":
        cfg = item_id[len("w4-") : item_id.rfind("-r")]
        return (f"config={cfg}",)
    if wave == "w5":
        pi = int(item_id.rsplit("-", 1)[1])
        a, b = W5_PAIRS[pi]
        return (f"pair={a}-vs-{b}",)
    return ()


def _cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a, "kappa needs equal nonempty label lists"
    labels = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(labels)}
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ma = np.bincount([idx[x] for x in a], minlength=len(labels)) / n
    mb = np.bincount([idx[x] for x in b], minlength=len(labels)) / n
    pe = float(np.dot(ma, mb))
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


def phase_w7(args) -> None:
    """Calibration: 200 items each from W3/W4/W5 re-judged with the project judge
    (claude-sonnet-4-5-20250929), same prompts; raw agreement + kappa per instrument."""
    p = _paths(args)
    g2 = _require_g2(args, p)
    t = _t2552()
    parents = {
        "w3": (compose_w3(args, p, g2), W3_SYSTEM),
        "w4": (compose_w4(args, p, g2)[0], W4_SYSTEM),
        "w5": (compose_w5(args, p, g2)[0], W5_SYSTEM),
    }
    out: dict[str, dict] = {}
    for k, (wave, (arms, system)) in enumerate(sorted(parents.items())):
        # the SAME final overlay production reductions use — base raw + rule-28 sync
        # re-issues folded in (#2552 r2 codex w7-calibration)
        primary = _reload_per_item(p, wave, wave)
        items = [it for arm_items in arms.values() for it in arm_items]
        valid_items = [it for it in items if primary.get(it[0], {}).get("class") == "valid"]
        rng = np.random.default_rng([SEED, 7, k])
        n = min(W7_N_PER_INSTRUMENT, len(valid_items))
        pick = [
            valid_items[int(i)] for i in sorted(rng.choice(len(valid_items), size=n, replace=False))
        ]
        cal_wave = f"w7{wave}"
        per_cal = run_wave(
            args,
            p,
            g2,
            wave=cal_wave,
            base_wave=wave,
            arms={f"{wave}-cal": pick},
            system=system,
            judge_model=CAL_JUDGE_MODEL,
        )
        if args.dry_run:
            continue
        if per_cal is None:  # resumed: rebuild WITH the sync overlay (#2552 r2)
            per_cal = _reload_per_item(p, cal_wave, wave)
        la: list[str] = []
        lb: list[str] = []
        n_cal_drop = 0
        cells: dict[str, dict] = {}
        for item_id, _q in pick:
            prim = primary[item_id]
            rec = per_cal.get(item_id)
            ok = rec is not None and rec["class"] == "valid"
            for cell_key in _w7_cells(wave, item_id, prim):
                c = cells.setdefault(
                    cell_key, {"n_sampled": 0, "n_both_valid": 0, "la": [], "lb": []}
                )
                c["n_sampled"] += 1
                if ok:
                    c["n_both_valid"] += 1
                    c["la"].append(json.dumps(prim["value"]))
                    c["lb"].append(json.dumps(rec["value"]))
            if not ok:
                n_cal_drop += 1
                continue
            la.append(json.dumps(prim["value"]))
            lb.append(json.dumps(rec["value"]))
        n_agree = sum(1 for x, y in zip(la, lb, strict=True) if x == y)
        cell_out = {}
        for cell_key, c in sorted(cells.items()):
            nbv = c["n_both_valid"]
            k_agree = sum(1 for x, y in zip(c["la"], c["lb"], strict=True) if x == y)
            lo, hi = _wilson(k_agree, nbv)
            cell_out[cell_key] = {
                "n_sampled": c["n_sampled"],
                "n_both_valid": nbv,
                "drop_rate_cal": (1.0 - nbv / c["n_sampled"]) if c["n_sampled"] else None,
                "agreement": (k_agree / nbv) if nbv else None,
                "agreement_wilson_ci95": [lo, hi],
                "kappa": _cohen_kappa(c["la"], c["lb"]) if nbv >= 2 else None,
            }
        lo_all, hi_all = _wilson(n_agree, len(la))
        out[wave] = {
            "n_sampled": n,
            "n_both_valid": len(la),
            "n_cal_dropped": n_cal_drop,
            "drop_rate_cal": (n_cal_drop / n) if n else float("nan"),
            "raw_agreement": (n_agree / len(la)) if la else float("nan"),
            "agreement_wilson_ci95": [lo_all, hi_all],
            "kappa": _cohen_kappa(la, lb) if la else float("nan"),
            "cells": cell_out,
            "cal_judge_model": CAL_JUDGE_MODEL,
        }
    if args.dry_run:
        return
    t.C.write_json_atomic(
        p.agg / "w7_calibration.json",
        {
            "instruments": out,
            "n_per_instrument": W7_N_PER_INSTRUMENT,
            "note": "per-cell n is thin (plan Alt-N): descriptive per-instrument agreement",
            **as_metadata_dict(git_provenance(), phase="judge-w7"),
        },
    )
    logger.info("[w7] calibration written for %d instruments", len(out))


def phase_all(args) -> None:
    for name in (
        "prep",
        "pilot-w1",
        "w1",
        "w2",
        "pilot-w3",
        "w3",
        "pilot-w4",
        "w4",
        "pilot-w5",
        "w5",
        "w6",
        "w7",
    ):
        print(f"[all] phase {name}", flush=True)
        PHASES[name](args)
    if not (args.dry_run or args.smoke):
        t = _t2552()
        t.C.write_json_atomic(
            Path(args.out_root) / "w_all_done.json",
            {
                "phase": "p2",
                "status": "done",
                "waves": ["w1", "w2", "w3", "w4", "w5", "w6", "w7"],
                **as_metadata_dict(git_provenance(), phase="judge-all"),
            },
        )


def phase_smoke_probes(args) -> None:
    """Tiny live SYNC probe per INSTRUMENT (the unit-3 smoke leg): w1-TA, w1-pt and
    w2 probe real composed items; w3..w6 probe composed blocks built from the w1/w2
    probe OUTPUTS (real judge text through the real parsers — instrument shape, not
    science). Outputs divert under <out_root>/smoke; canonical paths never written."""
    assert args.smoke, "phase smoke-probes requires --smoke"
    p = _paths(args)
    # INPUTS (g2 decision, staged prep fetches, eval texts) live in the CANONICAL
    # production prep tree — prep runs without --smoke; only probe OUTPUTS divert
    # under <out_root>/smoke (run_wave/_reload_per_item keep the diverted `p`).
    p_in = _paths(SimpleNamespace(out_root=args.out_root, smoke=False))
    g2 = _require_g2(args, p_in)
    # smoke view: composition runs over the same eval-id subset prep fetched texts for
    g2 = {**g2, "eval_ids": list(g2["eval_ids"])[:SMOKE_N_EVAL_TEXTS]}
    ta, pt = _compose_w1_split(args, p_in, g2)
    results: dict[str, dict] = {}

    def probe(wave: str, base: str, arms: dict, system: str) -> dict[str, dict]:
        per = run_wave(args, p, g2, wave=f"smoke_{wave}", base_wave=base, arms=arms, system=system)
        if per is None and not args.dry_run:  # resumed probe: reload from saved raw
            per = _reload_per_item(p, f"smoke_{wave}", base)
        per = per or {}
        n_valid = sum(1 for r in per.values() if r["class"] == "valid")
        results[wave] = {"n_items": sum(len(v) for v in arms.values()), "n_valid": n_valid}
        assert args.dry_run or n_valid >= 1, f"[smoke] wave {wave}: zero valid probe draws"
        return per

    per_w1 = probe("w1", "w1", {f: v[:2] for f, v in ta.items() if v}, W1_SYSTEM_TA)
    per_w1pt = probe("w1pt", "w1", {k: v[:5] for k, v in pt.items()}, W1_SYSTEM_PT)
    per_w2 = probe("w2", "w2", {a: v[:5] for a, v in compose_w2(args, p_in, g2).items()}, W2_SYSTEM)
    if args.dry_run:
        logger.info("[smoke] dry-run: w3..w6 probes need live w1/w2 outputs; skipped")
        return
    descs = [
        rec["value"]
        for src in (per_w1, per_w1pt)
        for _i, rec in sorted(src.items())
        if rec["class"] == "valid"
    ]
    summ = next(rec["value"] for _i, rec in sorted(per_w2.items()) if rec["class"] == "valid")
    eval_ids = [int(x) for x in g2["eval_ids"]][:12]
    texts = _load_texts(p_in, np.asarray(eval_ids, np.int64))
    lines = [f"- {d}" for d in descs]
    assert len(lines) >= 2, "[smoke] need >=2 valid descriptions for w3..w6 probes"
    w3_items = [
        (
            f"w3s-{k}",
            f"FEATURE DESCRIPTION:\n{d}\n\nTOP EXAMPLE EXCERPTS:\n"
            f"### Excerpt 1\n{texts[eval_ids[k % len(eval_ids)]][:800]}\n",
        )
        for k, d in enumerate(descs[:3])
    ]
    pp = w4_presentation(eval_ids[0], eval_ids)
    w4_parts = ["FEATURE DESCRIPTIONS (derived from one of the candidate turns):", ""]
    w4_parts.extend(lines)
    w4_parts.append("")
    w4_parts.append("CANDIDATE TURNS:")
    for lab, cand in zip(W4_LABELS, pp["candidates"], strict=True):
        w4_parts.append(f"### {lab}\n{texts[cand][:W4_CANDIDATE_CAP]}\n")
    half = max(1, len(lines) // 2)
    w5_block = (
        "STRUCTURED SUMMARY OF THE TURN:\n"
        + _render_summary(summ)
        + "\n\nLIST 1:\n"
        + "\n".join(lines[:half])
        + "\n\nLIST 2:\n"
        + "\n".join(lines[half:] or lines[:half])
    )
    w6_parts = ["STRUCTURED SUMMARY OF THE TURN:", _render_summary(summ), ""]
    for lab in W6_LABELS:
        w6_parts.append(f"LIST {lab}:")
        w6_parts.extend(lines)
        w6_parts.append("")
    probe("w3", "w3", {"probe": w3_items}, W3_SYSTEM)
    probe("w4", "w4", {"probe": [("w4s-0", "\n".join(w4_parts))]}, W4_SYSTEM)
    probe("w5", "w5", {"probe": [("w5s-0", w5_block)]}, W5_SYSTEM)
    probe("w6", "w6", {"probe": [("w6s-0", "\n".join(w6_parts))]}, W6_SYSTEM)
    t = _t2552()
    t.C.write_json_atomic(p.work / "smoke_probe_report.json", results)
    logger.info("[smoke] probes ok: %s", {k: v["n_valid"] for k, v in results.items()})


PHASES = {
    "prep": phase_prep,
    "pilot-w1": phase_pilot_w1,
    "w1": phase_w1,
    "w2": phase_w2,
    "pilot-w3": phase_pilot_w3,
    "w3": phase_w3,
    "pilot-w4": phase_pilot_w4,
    "w4": phase_w4,
    "pilot-w5": phase_pilot_w5,
    "w5": phase_w5,
    "w6": phase_w6,
    "w7": phase_w7,
    "all": phase_all,
    "smoke-probes": phase_smoke_probes,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wave", choices=sorted(PHASES), help="phase to run")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_2552" / "judge",
        help="work root (plan §9 sentinel lives here: w_all_done.json)",
    )
    ap.add_argument("--hf-prefix", default="issue2552_turnsae")
    ap.add_argument(
        "--dry-run", action="store_true", help="compose prompts + routing check; ZERO API calls"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny live SYNC probes; outputs divert under <out_root>/smoke",
    )
    ap.add_argument("--skip-upload", action="store_true", help="local-only (loud)")
    ap.add_argument(
        "--allow-below-floor",
        action="store_true",
        help="waive the rule-29 frac_items_complete HALT after drop-class triage "
        "(recorded in judge_meta as below_floor_waiver)",
    )
    ap.add_argument(
        "--max-chunks", type=int, default=0, help="0 = all 1,920 rollout chunks (production)"
    )
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attr + helper-bind check, execute deferred imports, exit 0",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(" ".join(sorted(PHASES)))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _t2552()  # execute the deferred torch-bearing import (smoke-contract Axis 1)
        print("[import-check] ok: argcheck + deferred issue2552_turnsae_der import")
        raise SystemExit(0)
    assert args.wave, "--wave is required (or --list-phases / --import-check)"
    args.out_root.mkdir(parents=True, exist_ok=True)
    PHASES[args.wave](args)
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
