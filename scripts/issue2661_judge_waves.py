"""Issue #2661 — VM-side Batch-API judge waves for the CONTEXT-side flat SAE.

Adapted from scripts/issue2552_judge_waves.py @ branch issue-2552 cb39df3ce1c
(dispatch envelope, draw classification, rule-28 sync re-issue, rule-29
completeness floor, MF-D pilots, W4 presentation determinism — VERBATIM where
possible; every wave retargeted to USER-MESSAGE evidence). Differences from the
parent, per the task body (## Provenance):

  - Judge model: claude-opus-5 (user override of the pinned project judge),
    passed EXPLICITLY per call site; NO assistant prefill anywhere (400 on this
    model); Batch route pinned via threshold_base=0.
  - ONE description family (the fresh ctx SAE). W1 evidence blocks state the
    text is the USER'S MESSAGE and carry non-activating negatives.
  - W2: Der Appendix D 24-field schema VERBATIM, wording adapted from
    "assistant turn" to "user message".
  - W4: 10-way matching of the ctx top-100 description list against the true
    prompt + 9 distractor prompts (fixed candidate sets, seed 2661).
  - W3 category assignment, W5 pairwise, W6 5-way: OUT (no comparator; W3 cut
    as not-cheap — the edge dashboard's topic-vs-behavior split reads the W1
    descriptions directly).
  - NEW `estimate` phase: counts calls + tokens + dollars per wave from the
    mined need-set BEFORE any dispatch (Opus 5 batch $2.50/M in, $12.50/M out),
    writes eval_results/issue_2661/judge_estimate.json, and every production
    wave REFUSES to dispatch when the upper-bound total exceeds --budget-usd
    (default 300; rc=9).
  - Pilots (>=51 draws, zero truncation, parse-fail < 2%, api-refusal < 0.10)
    gate w1, w2 AND w4 (the task body pins pilots before EACH production wave;
    the parent exempted sub-5k w2).

Phases: prep -> estimate -> pilot-w1 -> w1 -> pilot-w2 -> w2 -> pilot-w4 -> w4
(-> all). --dry-run composes prompts + routing with ZERO API calls.
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
logger = logging.getLogger("issue2661.judge")

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── instrument constants (task body ## Provenance) ───────────────────────────────
JUDGE_MODEL = "claude-opus-5"  # noqa: judge-model-pin -- task-body ## Provenance: "Judge [user-answer, overrides pinned project judge]: claude-opus-5"
WAVE_THRESHOLD_BASE = 0  # pins the Batch route for EVERY wave
SEED = 2661
FRAC_ITEMS_FLOOR = 0.95  # rule-29 per-arm completeness floor (#2552 parity)
MAX_TOKENS = {"w1": 1024, "w2": 2048, "w4": 1024}
EVAL_TURN_TEXT_CAP = 4_000  # chars: the summarized USER MESSAGE (W2)
W4_CANDIDATE_CAP = 1_500  # chars per W4 candidate prompt (10 per item)
W2_MIN_FIELDS = 20  # W2 validity floor: >= 20 of 24 fields present + non-empty
PILOT_MIN_EFFECTIVE = 51  # floor(1/0.02)+1 (#2124)
PILOT_PARSE_FAIL_THRESHOLD = 0.02
PILOT_API_REFUSAL_THRESHOLD = 0.10
RC_PILOT_FAIL = 7
RC_FLOOR_FAIL = 8
RC_BUDGET_FAIL = 9  # estimate-gate refusal (this driver's addition)
W4_LABELS = tuple("ABCDEFGHIJ")
_ITEM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,53}$")

# Opus 5 Batch pricing (task brief literals) + the estimate token approximation
BATCH_USD_PER_MTOK_IN = 2.50
BATCH_USD_PER_MTOK_OUT = 12.50
CHARS_PER_TOKEN = 3.5  # conservative chars->tokens divisor (recorded in the estimate)
EXPECTED_OUT_FRAC = 0.5  # "expected" spend variant; the GATE binds the upper bound
# PRE-DISPATCH spend gate only (task-body pin: 'refuse to dispatch above a
# --budget-usd cap'). NEVER a mid-run kill — the estimate gate runs BEFORE any
# dispatch; an in-flight wave is never aborted on cumulative spend (the
# tests/test_no_dollar_budget_caps.py invariant's incident class, #356).
SPEND_GATE_DEFAULT_USD = 300.0

REGIME_PINS = PROJECT_ROOT / "eval_results" / "issue_2661" / "regime_pins.json"
ESTIMATE_PATH = PROJECT_ROOT / "eval_results" / "issue_2661" / "judge_estimate.json"
HF_PREFIX_DEFAULT = "issue2661_flatsae"

# ── Appendix D schema (VERBATIM from issue2552_judge_waves.py, which fetched it
#    verbatim from Der et al. arXiv 2606.28548 Appendix D) ────────────────────────
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
assert len(APP_D_FIELDS) == 24 and len(APP_D_SCHEMA) == 5


def _schema_block() -> str:
    lines: list[str] = []
    for cat, fields in APP_D_SCHEMA:
        lines.append(f"{cat}:")
        for f, d in fields:
            lines.append(f"  - {f}: {d}")
    return "\n".join(lines)


# ── wave system prompts (USER-MESSAGE evidence; NO assistant prefill anywhere) ───
_JSON_ONLY = "Output ONLY a single JSON object and nothing else."

W1_SYSTEM_CTX = (
    "You are labeling features of a sparse autoencoder trained on language-model"
    " activations captured at the last token of the USER'S MESSAGE (the context side,"
    " before the assistant answers). You will be shown the user messages on which ONE"
    " feature activates most strongly, each with its activation value, followed by"
    " non-activating negative user messages. Write a short description (one or two"
    " sentences) of what the feature captures — the common content, style, or function"
    " of the USER MESSAGES it activates on. "
    + _JSON_ONLY
    + ' Use the form {"description": "<your description>"}.'
)
W2_SYSTEM = (
    "You will be shown one user message sent to an assistant. Produce a structured"
    " summary of it using EXACTLY this schema — populate each of the 24 fields with a"
    " short description of THIS user message (a phrase or one sentence per field); do"
    " not quote long spans of the message."
    "\n\nSCHEMA (5 categories, 24 fields):\n"
    + _schema_block()
    + "\n\n"
    + _JSON_ONLY
    + " Use a single flat JSON object whose keys are exactly the 24 field names above"
    " and whose values are short strings."
)
W4_SYSTEM = (
    "You will be shown a list of feature descriptions that were derived from ONE user"
    " message, followed by 10 candidate user messages labeled A through J. Exactly one"
    " candidate is the message the descriptions were derived from. Decide which. Reason"
    " briefly first, then answer. "
    + _JSON_ONLY
    + ' Use the form {"reason": "<brief reasoning>", "choice": "<one letter A-J>"}.'
)
WAVE_SYSTEMS = {"w1": W1_SYSTEM_CTX, "w2": W2_SYSTEM, "w4": W4_SYSTEM}


def _user_msg(question: str, completion: str) -> str:
    """format_user_msg for every wave (VERBATIM #2552): the composed content block
    IS the user message; instructions live in the SYSTEM prompt; ``completion``
    is unused by design (n_draws=1, answer slot empty)."""
    del completion
    return question


# ── lazy heavy import (unit 1; executed by --import-check) ───────────────────────


@functools.cache
def _u2661():
    import issue2661_flat_ctx_sae as u

    return u


def _sha_ids(ids: np.ndarray) -> str:
    return _u2661()._sha_ids(np.asarray(ids, np.int64))


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ── work-root layout ─────────────────────────────────────────────────────────────


def _paths(args) -> SimpleNamespace:
    work = args.out_root if not args.smoke else args.out_root / "smoke"
    agg = (
        PROJECT_ROOT / "eval_results" / "issue_2661" / "judge_aggregates"
        if not (args.smoke or args.dry_run)
        else work / "judge_aggregates"
    )
    work.mkdir(parents=True, exist_ok=True)
    agg.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(work=work, agg=agg)


def _pins() -> dict:
    assert REGIME_PINS.exists(), f"regime_pins.json missing at {REGIME_PINS}"
    return json.loads(REGIME_PINS.read_text())


def _resolve_pod_revision(args) -> str:
    """Pod uploads land AFTER the pin record, so the issue2661_flatsae prefix is
    resolved at fetch time to the CURRENT data-repo HEAD (#2552 r2
    p4-future-revision convention) and recorded in inputs_manifest.json."""
    u = _u2661()
    return u._resolve_repo_revision(None, "issue2661_flatsae HEAD resolve")


# ── wave-specific parsers (rule-27 round-trip-tested; VERBATIM #2552 w1/w2/w4) ───


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


def parse_w4(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    c = payload.get("choice")
    if isinstance(c, str):
        c = c.strip().strip("()").strip().upper()
        if c in W4_LABELS:
            return c
    return None


WAVE_PARSERS = {"w1": parse_w1, "w2": parse_w2, "w4": parse_w4}


# ── draw classification + reduce (VERBATIM #2552) ────────────────────────────────


def classify_draw(d: object, wave_parser) -> tuple[str, object]:
    """'transport' | 'api_refusal' | 'truncation' | 'parse_fail' | 'valid'."""
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
    per_item: dict[str, dict] = {}
    rank = {"valid": 0, "parse_fail": 1, "truncation": 2, "api_refusal": 3, "transport": 4}
    for custom_id, d in all_scores.items():
        item_id = custom_id.split("__")[0]
        cls, val = classify_draw(d, wave_parser)
        cur = per_item.get(item_id)
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


def _load_all_scores(save_raw: Path) -> dict[str, object]:
    doc = json.loads(save_raw.read_text())
    return doc.get("all_scores", {})


def _reload_per_item(p, wave: str, base: str) -> dict[str, dict]:
    """Rebuild a done wave's per-item reduce from persisted raw + sync overlay
    (VERBATIM #2552 resume path; zero API calls)."""
    parser = WAVE_PARSERS[base]
    raw = p.work / "raw" / wave / f"judge_raw_{wave}.json"
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


# ── dispatch + raw persistence (VERBATIM #2552, judge model explicit) ────────────


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
    n_draws=1; the empty completion is unused; NO assistant prefill on this path."""
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


def _upload_raw(args, p, wave: str, stage_files: list[Path]) -> None:
    """Raw judge outputs -> HF <hf_prefix>/raw_completions/judge/<wave>/ BEFORE
    any reduction consumes them (VERBATIM #2552)."""
    if args.skip_upload or args.smoke or args.dry_run:
        logger.warning("[%s] raw upload SKIPPED (skip_upload/smoke/dry_run) — loud", wave)
        return
    import shutil

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    u = _u2661()
    repo = u.C.HF_DATA_REPO
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
    u = _u2661()
    rows = [{"custom_id": cid, "result": d} for cid, d in sorted(all_scores.items())]
    base = p.work / "raw" / wave / f"judge_draws_{wave}_{tag}.jsonl"
    return u._jsonl_write_sharded(base, rows)


# ── the generic wave runner (VERBATIM #2552 run_wave; single-family trim) ─────────


def _wave_regime_key(args, manifest: dict, wave: str, system: str, judge_model: str) -> dict:
    return {
        "wave": wave,
        "eval_ids_sha256": manifest["eval_ids_sha256"],
        "need_set_sha256": manifest["need_set_sha256"],
        "judge_model": judge_model,
        "max_tokens": MAX_TOKENS[wave],
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
        assert "below_floor_arms" in doc, (
            f"judge_meta_{wave}.json predates the rule-29 completeness-floor gate — "
            "quarantine it and re-run the wave"
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


def _require_estimate(args) -> None:
    """The estimate gate: every PRODUCTION wave dispatch requires a committed
    judge_estimate.json whose upper-bound total sits under --budget-usd."""
    if args.dry_run or args.smoke:
        return
    assert ESTIMATE_PATH.exists(), (
        f"judge_estimate.json missing at {ESTIMATE_PATH} — run --wave estimate first"
    )
    doc = json.loads(ESTIMATE_PATH.read_text())
    total = float(doc["total"]["usd_upper_bound"])
    if total > float(args.budget_usd):
        print(
            f"[estimate-gate] REFUSED: upper-bound ${total:.2f} > budget "
            f"${float(args.budget_usd):.2f} (re-run --wave estimate after descoping, or "
            "raise --budget-usd deliberately)",
            flush=True,
        )
        raise SystemExit(RC_BUDGET_FAIL)


def run_wave(
    args,
    p,
    manifest: dict,
    *,
    wave: str,
    arms: dict[str, list[tuple[str, str]]],
    system: str,
    judge_model: str = JUDGE_MODEL,
    pilot_label: str | None = None,
) -> dict[str, dict] | None:
    """Dispatch -> persist raw -> reduce -> rule-28 sync re-issue -> merge -> meta
    (VERBATIM #2552 run_wave, single instrument family per wave)."""
    max_tokens = MAX_TOKENS[wave]
    parser = WAVE_PARSERS[wave]
    regime_key = _wave_regime_key(args, manifest, wave, system, judge_model)
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
        _u2661().C.write_json_atomic(
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
    _require_estimate(args)
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
    _upload_raw(args, p, wave, [save_raw, *shards])  # BEFORE reduction
    per_item = reduce_all_scores(all_scores, parser)
    missing = sorted({i for i, _q in items} - set(per_item))
    if missing:
        logger.warning(
            "[%s] %d expected items MISSING from persisted results — "
            "transport-lost, re-issued (rule 24)",
            wave,
            len(missing),
        )
    censored = [
        i for i, rec in per_item.items() if rec["class"] in ("api_refusal", "transport")
    ] + missing
    n_sync = 0
    if censored and not args.smoke:
        logger.warning("[%s] rule-28 sync re-issue: %d censored items", wave, len(censored))
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
        _upload_raw(args, p, wave, [save_raw2, *shards2])
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
            "[%s] BELOW frac_items_complete floor after re-issue: %s (rule 29)", wave, below
        )
        if not (args.smoke or args.allow_below_floor):
            _u2661().C.write_json_atomic(
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
                f"[{wave}] HALT: arms below the {FRAC_ITEMS_FLOOR} floor: {below} — "
                f"triage drop classes (rule 29); --allow-below-floor records a waiver",
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
        "wall_s": round(time.time() - t0, 1),
        **as_metadata_dict(git_provenance(), phase=f"judge-{wave}"),
    }
    (p.agg / f"floor_fail_{wave}.json").unlink(missing_ok=True)
    _u2661().C.write_json_atomic(p.agg / f"judge_meta_{wave}.json", meta)
    logger.info(
        "[%s] done: %d items, sync_reissue_valid=%d, below_floor=%s, wall=%.0fs",
        wave,
        n_calls,
        n_sync,
        below,
        time.time() - t0,
    )
    return per_item


# ── MF-D pilots (VERBATIM #2552 run_pilot; gates w1 AND w2 AND w4) ────────────────


def _pilot_sample(
    arms: dict[str, list[tuple[str, str]]], *, allow_small: bool = False
) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for k, (arm, items) in enumerate(sorted(arms.items())):
        if not allow_small:
            assert len(items) >= PILOT_MIN_EFFECTIVE, (
                f"pilot arm {arm} holds {len(items)} items < the {PILOT_MIN_EFFECTIVE}-draw "
                "floor (#2124 satisfiability)"
            )
        take = min(PILOT_MIN_EFFECTIVE, len(items))
        rng = np.random.default_rng([SEED, 26, k])
        idx = rng.choice(len(items), size=take, replace=False)
        out[arm] = [items[int(i)] for i in sorted(idx)]
    return out


def run_pilot(
    args,
    p,
    manifest: dict,
    *,
    wave: str,
    arms: dict[str, list[tuple[str, str]]],
    system: str,
    label: str | None = None,
) -> None:
    """(a) zero max_tokens truncation; (b) per-arm parse-fail < 2% under the
    wave's own parser; (c) batch-route parity + fresh cache; (d) api-refusal
    < 0.10; >= 51 effective draws/arm."""
    label = label or wave
    report_path = p.agg / f"pilot_report_{label}.json"
    if report_path.exists() and not args.dry_run:
        prior = json.loads(report_path.read_text())
        if prior.get("verdict") == "PASS":
            logger.info("[pilot-%s] resume: prior PASS report present; skip", label)
            return
        logger.warning("[pilot-%s] prior verdict=%s — fresh attempt", label, prior.get("verdict"))
    if not args.dry_run:
        _require_estimate(args)
    max_tokens = MAX_TOKENS[wave]
    parser = WAVE_PARSERS[wave]
    wave_n_calls = sum(len(v) for v in arms.values())
    pilot_arms = _pilot_sample(arms, allow_small=bool(args.smoke or args.dry_run))
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
        "attempt_root": str(root),
        **as_metadata_dict(git_provenance(), phase=f"judge-pilot-{label}"),
    }
    _u2661().C.write_json_atomic(report_path, report)
    logger.info("[pilot-%s] verdict=%s (%d arms)", label, verdict, len(pilot_arms))
    if verdict != "PASS":
        print(f"[pilot-{label}] FAIL: {failures}", flush=True)
        raise SystemExit(RC_PILOT_FAIL)


# ── prep: pod-artifact fetch + eval-prompt texts ─────────────────────────────────


def _inputs_dir(p) -> Path:
    d = p.work / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(p) -> Path:
    return p.work / "inputs_manifest.json"


def _list_pod_files(revision: str, hf_prefix: str, leaf: str) -> list[str]:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    u = _u2661()
    # HUB_VERIFY_RETRY_EXEMPT: the whole listing is wrapped in hub.retry_transient below
    tree = hub.retry_transient(
        lambda: list(
            HfApi().list_repo_tree(
                u.C.HF_DATA_REPO,
                repo_type="dataset",
                revision=revision,
                path_in_repo=f"{hf_prefix}/{leaf}",
            )
        ),
        what=f"list {hf_prefix}/{leaf}",
    )
    return [t.path for t in tree]


def _stage_pod_artifacts(args, p) -> dict:
    """Stage the pod outputs this driver consumes: eval lists (ctx), mining
    jsonls, need_set, top_pairs, receipts. --pod-out-root reads a LOCAL pod
    out-root (smoke/dev path); default fetches HF at the resolved revision."""
    u = _u2661()
    dst = _inputs_dir(p)
    doc: dict = {}
    if args.pod_out_root is not None:
        src = Path(args.pod_out_root)
        import shutil

        pairs = [
            (src / "eval_lists" / "feature_lists_2000turns.json", dst),
            *[(f, dst) for f in sorted((src / "eval_lists").glob("feature_lists_ctx*.jsonl"))],
            *[(f, dst) for f in sorted((src / "mining").glob("top25_ctx*.jsonl"))],
            (src / "mining" / "need_set.json", dst),
            (src / "edges" / "top_pairs.json", dst),
            (src / "edges" / "receipts_answer_features.json", dst),
        ]
        for f, d in pairs:
            assert f.exists(), f"[prep] pod artifact missing under --pod-out-root: {f}"
            shutil.copy2(f, d / f.name)
        doc["source"] = {"pod_out_root": str(src)}
        return doc
    rev = _resolve_pod_revision(args)
    doc["source"] = {"hf_revision": rev, "hf_prefix": args.hf_prefix}
    for leaf, pattern in (
        ("analysis_tensors/eval_lists", r"feature_lists_(2000turns\.json|ctx.*\.jsonl)$"),
        ("raw_completions/mining", r"top25_ctx.*\.jsonl$"),
    ):
        for path in _list_pod_files(rev, args.hf_prefix, leaf):
            if re.search(pattern, path):
                got = u._hf_fetch(path, dst / "_hf", rev)
                (dst / Path(path).name).write_bytes(Path(got).read_bytes())
    for path in (
        "analysis_tensors/need_set.json",
        "analysis_tensors/edges/top_pairs.json",
        "analysis_tensors/edges/receipts_answer_features.json",
    ):
        got = u._hf_fetch(f"{args.hf_prefix}/{path}", dst / "_hf", rev)
        (dst / Path(path).name).write_bytes(Path(got).read_bytes())
    return doc


def _load_ctx_lists(p) -> dict[int, list[int]]:
    """{row_id: judged top-100 ctx feature ids} from the staged sharded lists."""
    dst = _inputs_dir(p)
    idx = json.loads((dst / "feature_lists_2000turns.json").read_text())
    cfg = idx["configs"]["ctx"]
    out: dict[int, list[int]] = {}
    for fname in cfg["files"]:
        with (dst / fname).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    out[int(row["row_id"])] = [int(f) for f, _v in row["judged_top100"]]
    assert len(out) == int(cfg["n_turns"]), (len(out), cfg["n_turns"])
    return out


def _mining_records(p) -> list[dict]:
    dst = _inputs_dir(p)
    recs: list[dict] = []
    for f in sorted(dst.glob("top25_ctx*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    assert recs, "[prep] no mining records staged (top25_ctx*.jsonl)"
    return recs


def _grouped_mining(p) -> dict[int, dict[str, list[dict]]]:
    by_feat: dict[int, dict[str, list[dict]]] = {}
    for rec in _mining_records(p):
        kind = "positive" if rec.get("kind") == "positive" else "negative"
        by_feat.setdefault(int(rec["feat_id"]), {"positive": [], "negative": []})[kind].append(rec)
    for d in by_feat.values():
        d["positive"].sort(key=lambda r: int(r["rank"]))
        d["negative"].sort(key=lambda r: int(r["rank"]))
    return by_feat


def _texts_path(p) -> Path:
    return p.work / "eval_prompt_texts.jsonl"


def _fetch_eval_texts(args, p, eval_ids: np.ndarray, row_ci: np.ndarray) -> None:
    """Stream the pinned rollout chunks and persist each eval turn's USER-PROMPT
    text (capped) — the #2552 _fetch_eval_texts shape, PROMPT field instead of
    response. Checkpointed per row; text never logged."""
    u = _u2661()
    out = _texts_path(p)
    state_path = p.work / "eval_texts_state.json"
    fingerprint = {
        "eval_ids_sha256": _sha_ids(eval_ids),
        "text_cap": EVAL_TURN_TEXT_CAP,
        "field": "prompt",
    }
    if state_path.exists():
        prev = json.loads(state_path.read_text())
        assert prev == fingerprint, (
            f"eval_texts_state.json regime drift: {prev} != {fingerprint} — move the stale "
            "texts aside rather than silently reusing another regime's"
        )
    else:
        u.C.write_json_atomic(state_path, fingerprint)
    have: set[int] = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    have.add(int(json.loads(line)["row_id"]))
    todo = [int(r) for r in eval_ids if int(r) not in have]
    if not todo:
        logger.info("[texts] resume: all %d eval prompts present", len(eval_ids))
        return
    needed_ci = {int(row_ci[r]): int(r) for r in todo}
    assert len(needed_ci) == len(todo) and all(c >= 0 for c in needed_ci), (
        "eval ids must be text-resolvable (ci>=0)"
    )
    ns = SimpleNamespace(out_root=p.work, max_chunks=args.max_chunks)
    n0 = len(have)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for k, (row_idx, ci, prompt, _response) in enumerate(
            u._iter_rows_pinned(ns, needed_ci, tag="eval_texts")
        ):
            rec = {
                "row_id": int(row_idx),
                "ci": int(ci),
                "text": prompt[:EVAL_TURN_TEXT_CAP],
                "n_chars_orig": len(prompt),
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
        f"[texts] recovered {n_final}/{len(eval_ids)} eval prompts — the 2,000/2,000 "
        "eval-row/text join is violated"
    )
    logger.info("[texts] complete: %d prompts at cap %d chars", n_final, EVAL_TURN_TEXT_CAP)


def _load_texts(p, eval_ids: np.ndarray) -> dict[int, str]:
    out: dict[int, str] = {}
    with _texts_path(p).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out[int(rec["row_id"])] = rec["text"]
    missing = [int(r) for r in eval_ids if int(r) not in out]
    assert not missing, f"eval prompts missing for {len(missing)} rows — rerun --wave prep"
    return out


def _require_manifest(p) -> dict:
    path = _manifest_path(p)
    assert path.exists(), "inputs_manifest.json missing — run --wave prep first"
    return json.loads(path.read_text())


def phase_prep(args) -> None:
    """Stage pod artifacts + fetch the 2,000 eval USER-PROMPT texts + write the
    inputs manifest (eval/need identity shas the wave regime keys bind on)."""
    p = _paths(args)
    u = _u2661()
    src_doc = _stage_pod_artifacts(args, p)
    lists = _load_ctx_lists(p)
    eval_ids = np.asarray(sorted(lists), np.int64)
    need_doc = json.loads((_inputs_dir(p) / "need_set.json").read_text())
    need_ids = np.asarray(sorted(int(x) for x in need_doc["need_ids"]), np.int64)
    # eval-prompt texts via the pinned chunk sweep (row_ci from the staged meta)
    ns = SimpleNamespace(scratch=u.T._stage_dir(SimpleNamespace(out_root=p.work)))
    ns.scratch.mkdir(parents=True, exist_ok=True)
    u.EL._stage_scratch_meta(ns)
    row_ci = np.load(ns.scratch / "row_ci.npy")
    if args.smoke:
        eval_ids = eval_ids[: min(len(eval_ids), 8)]
    _fetch_eval_texts(args, p, eval_ids, row_ci)
    manifest = {
        **src_doc,
        "n_eval": int(len(eval_ids)),
        "eval_ids": [int(x) for x in eval_ids],
        "eval_ids_sha256": _sha_ids(eval_ids),
        "n_need": int(len(need_ids)),
        "need_set_sha256": _sha_ids(need_ids),
        "need_set_provenance": {
            k: v for k, v in need_doc.items() if k not in ("need_ids", "metadata")
        },
        **as_metadata_dict(git_provenance(), phase="judge-prep"),
    }
    u.C.write_json_atomic(_manifest_path(p), manifest)
    logger.info("[prep] done: n_eval=%d n_need=%d", len(eval_ids), len(need_ids))


# ── item composition per wave ─────────────────────────────────────────────────────


def _w1_block_ctx(recs: dict[str, list[dict]]) -> str:
    parts = ["FEATURE EXAMPLES (top-activating user messages, activation shown):", ""]
    for k, r in enumerate(recs["positive"]):
        parts.append(f"### Example {k + 1} (activation={float(r['activation']):.4f})")
        parts.append(str(r["text"]))
        parts.append("")
    if recs["negative"]:
        parts.append("NON-ACTIVATING NEGATIVES (user messages where this feature does NOT fire):")
        parts.append("")
        for k, r in enumerate(recs["negative"]):
            parts.append(f"### Negative {k + 1}")
            parts.append(str(r["text"]))
            parts.append("")
    return "\n".join(parts)


def compose_w1(args, p, manifest: dict) -> dict[str, list[tuple[str, str]]]:
    """One arm (ctx family); item per need-set feature with >= 1 positive mining
    row (zero-firing need features are dropped + recorded, #2552 convention)."""
    mined = _grouped_mining(p)
    need_doc = json.loads((_inputs_dir(p) / "need_set.json").read_text())
    need = sorted(int(x) for x in need_doc["need_ids"])
    missing = sorted(f for f in need if f not in mined or not mined[f]["positive"])
    if missing:
        logger.warning(
            "[w1] %d need features have no positive mining rows — dropped, recorded", len(missing)
        )
    coverage = {
        "n_need": len(need),
        "n_described": len(need) - len(missing),
        "n_dropped_zero_mining": len(missing),
        "dropped_ids": missing,
    }
    if not (args.smoke or args.dry_run):
        _u2661().C.write_json_atomic(p.agg / "w1_mining_coverage.json", coverage)
    items = [
        (f"w1-ctx-f{feat}", _w1_block_ctx(mined[feat])) for feat in need if feat not in set(missing)
    ]
    if args.smoke:
        items = items[: max(2, min(len(items), 5))]
    return {"ctx": items}


def compose_w2(args, p, manifest: dict) -> dict[str, list[tuple[str, str]]]:
    eval_ids = np.asarray(manifest["eval_ids"], np.int64)
    texts = _load_texts(p, eval_ids)
    items = [
        (f"w2-r{int(r)}", f"USER MESSAGE (sent to an assistant):\n\n{texts[int(r)]}")
        for r in eval_ids
    ]
    return {"all_turns": items}


def _descriptions_ctx(p) -> dict[int, str]:
    path = p.agg / "descriptions_ctx.json"
    assert path.exists(), "descriptions_ctx.json missing — run --wave w1 first"
    doc = json.loads(path.read_text())
    return {int(k): v for k, v in doc["descriptions"].items()}


def w4_presentation(row_id: int, pool: list[int]) -> dict:
    """Deterministic per-turn candidate set + order (VERBATIM #2552, seed 2661).
    Production draws 9 distractors; a sub-10 smoke pool degrades to
    min(9, len(others)) so the tiny composed probe still runs."""
    others = sorted(int(x) for x in pool if int(x) != int(row_id))
    rng = np.random.default_rng([SEED, 4, int(row_id)])
    k = min(9, len(others))
    assert k >= 1, f"w4 needs >= 2 eval rows (row {row_id} has no distractor pool)"
    distractors = [int(x) for x in rng.choice(others, size=k, replace=False)]
    cands = [int(row_id), *distractors]
    order = [cands[i] for i in rng.permutation(k + 1)]
    gold_label = W4_LABELS[order.index(int(row_id))]
    return {"candidates": order, "gold_label": gold_label}


def compose_w4(
    args, p, manifest: dict
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, dict], dict]:
    eval_ids = [int(x) for x in manifest["eval_ids"]]
    texts = _load_texts(p, np.asarray(eval_ids, np.int64))
    turn_feats = _load_ctx_lists(p)
    desc = _descriptions_ctx(p)
    pres = {int(r): w4_presentation(int(r), eval_ids) for r in eval_ids}
    items: list[tuple[str, str]] = []
    row_meta: dict[str, dict] = {}
    for r in eval_ids:
        feats = turn_feats.get(r, [])
        lines = [f"- {desc[int(f)]}" for f in feats if int(f) in desc]
        n_missing = sum(1 for f in feats if int(f) not in desc)
        item_id = f"w4-ctx-r{r}"
        if not lines:
            row_meta[item_id] = {
                "row_id": r,
                "valid": False,
                "reason": "no_described_features",
                "n_missing_desc": n_missing,
            }
            continue
        pp = pres[r]
        parts = ["FEATURE DESCRIPTIONS (derived from one of the candidate user messages):", ""]
        parts.extend(lines)
        parts.append("")
        parts.append("CANDIDATE USER MESSAGES:")
        for lab, cand in zip(W4_LABELS[: len(pp["candidates"])], pp["candidates"], strict=True):
            parts.append(f"### {lab}\n{texts[cand][:W4_CANDIDATE_CAP]}\n")
        items.append((item_id, "\n".join(parts)))
        row_meta[item_id] = {
            "row_id": r,
            "valid": None,
            "gold": pp["gold_label"],
            "n_desc": len(lines),
            "n_missing_desc": n_missing,
        }
    if args.smoke:
        items = items[: max(2, min(len(items), 5))]
    return {"ctx": items}, row_meta, {str(k): v for k, v in pres.items()}


# ── estimate (NEW: the budget gate's input) ──────────────────────────────────────


def _est_wave(items: list[tuple[str, str]], system: str, wave: str) -> dict:
    n = len(items)
    in_chars = sum(len(q) for _i, q in items) + n * len(system)
    in_tok = in_chars / CHARS_PER_TOKEN
    out_cap = n * MAX_TOKENS[wave]
    return {
        "n_items": n,
        "input_chars": int(in_chars),
        "est_input_tokens": int(in_tok),
        "output_token_cap": int(out_cap),
        "usd_upper_bound": in_tok / 1e6 * BATCH_USD_PER_MTOK_IN
        + out_cap / 1e6 * BATCH_USD_PER_MTOK_OUT,
        "usd_expected": in_tok / 1e6 * BATCH_USD_PER_MTOK_IN
        + out_cap * EXPECTED_OUT_FRAC / 1e6 * BATCH_USD_PER_MTOK_OUT,
    }


def phase_estimate(args) -> None:
    """Count calls + tokens + dollars per wave from the REALIZED mined need-set
    BEFORE any dispatch. W4 composes against PLACEHOLDER description lines (W1
    has not run yet) sized by DESC_CHAR_EST — recorded in the JSON. Exits rc=9
    when the upper bound exceeds --budget-usd (the wave phases re-check)."""
    DESC_CHAR_EST = 120
    p = _paths(args)
    manifest = _require_manifest(p)
    w1 = compose_w1(args, p, manifest)
    w2 = compose_w2(args, p, manifest)
    eval_ids = [int(x) for x in manifest["eval_ids"]]
    texts = _load_texts(p, np.asarray(eval_ids, np.int64))
    turn_feats = _load_ctx_lists(p)
    w4_items: list[tuple[str, str]] = []
    for r in eval_ids:
        n_desc = len(turn_feats.get(r, []))
        desc_block = "\n".join(["- " + "x" * DESC_CHAR_EST] * max(1, n_desc))
        cand_block = "\n".join(
            f"### {lab}\n{texts[c][:W4_CANDIDATE_CAP]}\n"
            for lab, c in zip(W4_LABELS, [r] * 10, strict=True)
        )
        w4_items.append((f"w4-ctx-r{r}", desc_block + "\n" + cand_block))
    waves = {
        "w1": _est_wave([i for a in w1.values() for i in a], W1_SYSTEM_CTX, "w1"),
        "w2": _est_wave([i for a in w2.values() for i in a], W2_SYSTEM, "w2"),
        "w4": _est_wave(w4_items, W4_SYSTEM, "w4"),
    }
    # pilots: 51 items per wave at that wave's mean item size (all three piloted)
    pilots = {}
    for wname, west in waves.items():
        if west["n_items"] == 0:
            continue
        frac = min(1.0, PILOT_MIN_EFFECTIVE / west["n_items"])
        pilots[f"pilot_{wname}"] = {
            "n_items": min(PILOT_MIN_EFFECTIVE, west["n_items"]),
            "usd_upper_bound": west["usd_upper_bound"] * frac,
            "usd_expected": west["usd_expected"] * frac,
        }
    total_upper = sum(w["usd_upper_bound"] for w in waves.values()) + sum(
        w["usd_upper_bound"] for w in pilots.values()
    )
    total_expected = sum(w["usd_expected"] for w in waves.values()) + sum(
        w["usd_expected"] for w in pilots.values()
    )
    doc = {
        "judge_model": JUDGE_MODEL,
        "pricing_batch_usd_per_mtok": {"in": BATCH_USD_PER_MTOK_IN, "out": BATCH_USD_PER_MTOK_OUT},
        "chars_per_token_divisor": CHARS_PER_TOKEN,
        "expected_out_frac": EXPECTED_OUT_FRAC,
        "w4_desc_placeholder_chars": DESC_CHAR_EST,
        "waves": waves,
        "pilots": pilots,
        "total": {
            "usd_upper_bound": round(total_upper, 2),
            "usd_expected": round(total_expected, 2),
        },
        "budget_usd": float(args.budget_usd),
        "verdict": "UNDER_BUDGET" if total_upper <= float(args.budget_usd) else "OVER_BUDGET",
        "note": "upper bound assumes every call emits max_tokens; the dispatch gate binds "
        "on usd_upper_bound",
        **as_metadata_dict(git_provenance(), phase="judge-estimate"),
    }
    out_path = (
        ESTIMATE_PATH if not (args.smoke or args.dry_run) else (p.work / "judge_estimate.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _u2661().C.write_json_atomic(out_path, doc)
    print(
        f"[estimate] {json.dumps({'total': doc['total'], 'verdict': doc['verdict'], 'waves': {k: {'n': v['n_items'], 'usd_ub': round(v['usd_upper_bound'], 2)} for k, v in waves.items()}})}",
        flush=True,
    )
    if doc["verdict"] == "OVER_BUDGET":
        print(
            f"[estimate] OVER BUDGET: ${total_upper:.2f} > ${float(args.budget_usd):.2f} — "
            "waves will refuse to dispatch (rc=9)",
            flush=True,
        )
        raise SystemExit(RC_BUDGET_FAIL)


# ── wave phases (compose + run + aggregate) ──────────────────────────────────────


def phase_pilot_w1(args) -> None:
    p = _paths(args)
    manifest = _require_manifest(p)
    run_pilot(
        args,
        p,
        manifest,
        wave="w1",
        arms=compose_w1(args, p, manifest),
        system=W1_SYSTEM_CTX,
        label="w1",
    )


def phase_w1(args) -> None:
    p = _paths(args)
    manifest = _require_manifest(p)
    per = run_wave(
        args,
        p,
        manifest,
        wave="w1",
        arms=compose_w1(args, p, manifest),
        system=W1_SYSTEM_CTX,
        pilot_label="w1",
    )
    if args.dry_run:
        return
    if per is None:
        per = _reload_per_item(p, "w1", "w1")
    pref = "w1-ctx-f"
    descs = {
        int(i.removeprefix(pref)): rec["value"]
        for i, rec in per.items()
        if i.startswith(pref) and rec["class"] == "valid"
    }
    n_drop = sum(1 for i, rec in per.items() if i.startswith(pref) and rec["class"] != "valid")
    _u2661().C.write_json_atomic(
        p.agg / "descriptions_ctx.json",
        {
            "family": "ctx",
            "judge_model": JUDGE_MODEL,
            "n_valid": len(descs),
            "n_dropped": n_drop,
            "descriptions": {str(k): v for k, v in sorted(descs.items())},
        },
    )
    logger.info("[w1] ctx: %d descriptions (%d dropped)", len(descs), n_drop)


def phase_pilot_w2(args) -> None:
    p = _paths(args)
    manifest = _require_manifest(p)
    run_pilot(
        args,
        p,
        manifest,
        wave="w2",
        arms=compose_w2(args, p, manifest),
        system=W2_SYSTEM,
        label="w2",
    )


def phase_w2(args) -> None:
    p = _paths(args)
    manifest = _require_manifest(p)
    per = run_wave(
        args,
        p,
        manifest,
        wave="w2",
        arms=compose_w2(args, p, manifest),
        system=W2_SYSTEM,
        pilot_label="w2",
    )
    if args.dry_run:
        return
    if per is None:
        per = _reload_per_item(p, "w2", "w2")
    pref = "w2-r"
    summaries = {
        int(i.removeprefix(pref)): rec["value"]
        for i, rec in per.items()
        if i.startswith(pref) and rec["class"] == "valid"
    }
    n_drop = sum(1 for i, rec in per.items() if i.startswith(pref) and rec["class"] != "valid")
    _u2661().C.write_json_atomic(
        p.agg / f"summaries_{manifest['n_eval']}.json",
        {
            "n_valid": len(summaries),
            "n_dropped": n_drop,
            "min_fields": W2_MIN_FIELDS,
            "schema_fields": list(APP_D_FIELDS),
            "summaries": {str(k): v for k, v in sorted(summaries.items())},
        },
    )
    logger.info("[w2] %d summaries (%d dropped)", len(summaries), n_drop)


def phase_pilot_w4(args) -> None:
    p = _paths(args)
    manifest = _require_manifest(p)
    arms, _meta_rows, _pres = compose_w4(args, p, manifest)
    run_pilot(args, p, manifest, wave="w4", arms=arms, system=W4_SYSTEM, label="w4")


def phase_w4(args) -> None:
    p = _paths(args)
    manifest = _require_manifest(p)
    arms, row_meta, pres = compose_w4(args, p, manifest)
    per = run_wave(args, p, manifest, wave="w4", arms=arms, system=W4_SYSTEM, pilot_label="w4")
    if args.dry_run:
        return
    if per is None:
        per = _reload_per_item(p, "w4", "w4")
    u = _u2661()
    records = []
    n_correct = n_valid = 0
    for item_id, meta_row in sorted(row_meta.items()):
        rec = per.get(item_id)
        if meta_row.get("valid") is False or rec is None:
            records.append({**meta_row, "item_id": item_id, "class": "not_dispatched"})
            continue
        ok = rec["class"] == "valid"
        correct = bool(ok and rec["value"] == meta_row["gold"])
        n_valid += int(ok)
        n_correct += int(correct)
        records.append(
            {
                **meta_row,
                "item_id": item_id,
                "class": rec["class"],
                "choice": rec.get("value"),
                "correct": correct,
            }
        )
    acc = (n_correct / n_valid) if n_valid else float("nan")
    u.C.write_json_atomic(
        p.agg / "w4_matching.json",
        {
            "n_valid": n_valid,
            "n_correct": n_correct,
            "accuracy": acc,
            "wilson_ci": u.T._wilson(acc, n_valid) if n_valid else None,
            "chance": 0.1,
            "seed": SEED,
            "records": records,
            "presentation": pres,
        },
    )
    logger.info("[w4] acc=%.4f over %d valid items (chance 0.10)", acc, n_valid)


def phase_all(args) -> None:
    for name in ("prep", "estimate", "pilot-w1", "w1", "pilot-w2", "w2", "pilot-w4", "w4"):
        logger.info("[all] -> %s", name)
        PHASES[name](args)


PHASES = {
    "prep": phase_prep,
    "estimate": phase_estimate,
    "pilot-w1": phase_pilot_w1,
    "w1": phase_w1,
    "pilot-w2": phase_pilot_w2,
    "w2": phase_w2,
    "pilot-w4": phase_pilot_w4,
    "w4": phase_w4,
    "all": phase_all,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wave", choices=sorted(PHASES), required=False, help="phase to run")
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data" / "issue_2661" / "judge")
    ap.add_argument("--hf-prefix", default=HF_PREFIX_DEFAULT)
    ap.add_argument(
        "--pod-out-root",
        type=Path,
        default=None,
        help="read pod artifacts from a LOCAL out-root instead of HF",
    )
    ap.add_argument(
        "--budget-usd",
        type=float,
        default=SPEND_GATE_DEFAULT_USD,
        help="estimate-gate cap; waves refuse to dispatch above it",
    )
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
        help="rule-29 waiver after drop-class triage (recorded)",
    )
    ap.add_argument(
        "--max-chunks", type=int, default=0, help="0 = all 1,920 rollout chunks (production)"
    )
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="argparse-attr completeness + deferred imports, exit 0",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps({"registry": sorted(PHASES)}))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _u2661()  # unit-1 import resolves (torch chain)
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.upload_sharded import (  # noqa: F401
            upload_dir_sharded,
        )

        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    assert args.wave, "--wave is required (or --list-phases / --import-check)"
    logger.info(
        "[main] wave=%s out_root=%s dry_run=%s smoke=%s judge=%s",
        args.wave,
        args.out_root,
        args.dry_run,
        args.smoke,
        JUDGE_MODEL,
    )
    PHASES[args.wave](args)
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
