#!/usr/bin/env python3
"""Issue #2202 — Fable-5 synthesis (hypothesis generation) + Sonnet re-labeling wave.

VM-side driver (plan v2 §4 P3/P4). Phases:

- ``fable-digest``  P3a: build the population (FAIL-1 + matched non-failure
                    control with the pre-registered equalize-down fallback +
                    the 500-sample) and the two bounded Fable digest JSONLs
                    (Result 1: WORST-200 + stratified-300 of remaining
                    failures; Result 2: the sample rows), joining pod geometry
                    with the LOCAL #1482 text cache; upload digests to HF.
- ``fable-read``    P3b: production-shape pilot gate (1 real digest chunk at
                    the production prompt size + max_tokens; gates on a
                    non-empty text block, ``stop_reason != "max_tokens"``, and
                    a parseable modes schema — rc 25 designed halt), then
                    chunked ``claude-fable-5`` synthesis calls via
                    ``llm/api_dispatch.dispatch_calls`` (sync; ~40 chunk calls
                    + 1 consolidation call at 25 rows/chunk). Raw outputs +
                    per-chunk ``stop_reason`` persist VERBATIM; blank replies
                    and max_tokens truncations are HARD errors (never cached
                    as success). Content refusals (``stop_reason ==
                    "refusal"``, zero content blocks — ~deterministic, probed
                    at exact production shape) are FIRST-CLASS non-retried
                    outcomes: a refused chunk re-dispatches as single-row
                    items and rows that individually refuse are DROPPED and
                    recorded (``refusal_exclusions.json`` + the
                    ``[p3b] refusal exclusions:`` summary line; a refusing
                    pilot still halts rc 25 by design). Consolidation is
                    HIERARCHICAL (crash-fix 2: the single all-proposals call
                    refused at aggregate size — 1,297 proposals / 647,685
                    chars, while a 100-proposal subset PASSes at exact
                    production shape): stage 1 consolidates deterministic
                    contiguous batches of ``FABLE_CONSOL_BATCH`` proposals in
                    ONE dispatch call, a refused batch is half-split ONCE and
                    a still-refusing half is dropped-and-reported (the
                    ``consolidation_dropped`` section + the ``[p3b]
                    consolidation exclusions:`` line); stage 2 merges the
                    surviving stage-1 modes with the SAME instruction into
                    ≤ 10 snake_case modes with one-line decision rules
                    (``modes.json``, which carries the exclusion counts +
                    consolidation topology for the coverage caveat). Schema
                    parse failures and a refused/unparseable stage-2 stay
                    hard rc-25 halts (a schema failure is a bug, not
                    content). Fable never carries a countable claim.
- ``sonnet-pilot``  P4a: 5-request LIVE forced-batch probe through the SAME
                    request builder (the #763 mock-smoke gap), then the ~150
                    pilot at the exact production instrument against a FRESH
                    pilot checkpoint dir; gate = zero ``stop_reason ==
                    "max_tokens"`` AND per-arm parse-fail < 2% (rc 22 designed
                    halt on FAIL — the production wave never dispatches past
                    a failed pilot).
- ``sonnet-wave``   P4b: full population wave (~1,829 fail + ~1,829 control +
                    500 sample) — `claude-sonnet-4-5-20250929`, Batch API via
                    ``dispatch_judge_items``, max_tokens=2048, temperature =
                    API default (1.0, n_draws=1 — the parent instrument's own
                    setting), drop-never-coerce with per-arm counts split
                    content vs transport.
- ``retest``        P4c: 200-item test-retest with ``rt_``-prefixed custom ids
                    against a SEPARATE checkpoint dir (cache-identity per the
                    parent's :303 convention); κ per mode; κ < 0.6 (or
                    non-finite) modes DEMOTED to report-only; final
                    ``judge_labels_2202/labels.json`` + HF mirror.

Judge instrument settings mirror the parent #1738 instrument
(`issue1738_characterize.phase_judge`): API-default temperature, n_draws=1,
`rt_` retest prefix — so the κ range is commensurable with the parent's
0.79–0.98 / κ-0.6 convention. Refusal-safety: corpus text rides ONLY in API
payloads, the gitignored digests under data/, and the dashboard layer — it is
never printed or logged by this driver.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1482_analysis as A82  # noqa: E402  (_cohens_kappa; judge-model pin)
import issue2202_failchar as FC  # noqa: E402  (constants + shard/meta/json helpers)
import numpy as np  # noqa: E402

from explore_persona_space.eval.batch_judge import is_transport_error_dict  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2202_labels")

JUDGE_MODEL = A82.JUDGE_MODEL  # claude-sonnet-4-5-20250929 (project pin)
JUDGE_MAX_TOKENS = 2_048  # multi-field reason-then-score JSON rubric floor (rule 23)
FABLE_MODEL = "claude-fable-5"  # hypothesis generation ONLY (task-body lock)
# On Fable, reasoning tokens COUNT AGAINST max_tokens (thinking is always on and
# cannot be disabled or separately budgeted — budget_tokens 400s on this model).
# The prior 8_000 cap was exhausted by reasoning before any text block on 7/10
# digest chunks (empty replies recorded as success — the fable-digest-rerun
# incident). 32_000 = 4x the old cap on prompts 4x smaller (chunk rows 100->25);
# a cap is not a spend (llm-judging rule 23) — only realized tokens bill. The
# production-shape pilot gate below validates the sizing live before the wave.
FABLE_MAX_TOKENS = 32_000
# max_tokens > ~21,333 on the SDK's NON-streaming path requires an explicit
# per-request timeout: anthropic 0.88.0 `_calculate_nonstreaming_timeout` raises
# ValueError when expected time (3600s * max_tokens / 128_000) exceeds 10 min
# and no explicit timeout is given (api_dispatch builds its clients without one).
FABLE_REQUEST_TIMEOUT_S = 3_600.0
MODE_CAP = 10  # hard cap (plan: expected M <= 8)
# Hierarchical consolidation (fable-digest-rerun crash-fix 2). The single
# all-proposals consolidation call refuses at aggregate size (measured live at
# exact production shape: 1,297 proposals / 647,685 chars ->
# stop_reason=="refusal", empty text; a 100-proposal subset / 50,789 chars ->
# end_turn, 10 modes). Refusal is aggregate-triggered — prompt-framing
# workarounds were probed live and closed — so consolidation runs in stages:
# stage-1 batches of FABLE_CONSOL_BATCH proposals (all batches in ONE dispatch
# call), half-split retry for a refused batch (a still-refusing half is
# dropped-and-reported), then ONE stage-2 final merge over the surviving
# stage-1 modes.
FABLE_CONSOL_BATCH = 100
# One conditional extra stage-1 pass (NO generic recursion machinery) fires
# only when the stage-2 input would exceed this many proposals-equivalent;
# expected stage-2 input is ~13 batches x <= MODE_CAP modes ~ 130, far below.
FABLE_CONSOL_STAGE2_MAX = 3_000
KAPPA_DEMOTE = 0.6
PILOT_N = 150
RETEST_N = 200
DIGEST_WORST = 200
DIGEST_STRAT = 300
CAP_LAST_USER = 1_200  # locked excerpt caps (task body / #1738 instrument)
CAP_HISTORY = 800
CAP_RESPONSE = 1_000
CAP_CONFUSER = 400
RC_PILOT = 22  # designed-halt rc for the judge pilot gate (never bare 1)
RC_FABLE = 25  # designed-halt rc for a Fable probe/consolidation failure
DEFAULT_TEXT_CACHE = str(
    PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch" / "judge_texts.jsonl"
)
LABELS_1738_REL = "eval_results/issue_1738/judge_labels/labels.json"


# ── shared loaders ────────────────────────────────────────────────────────────────


def cap_text(s: str, n: int) -> str:
    """Excerpt cap with the inline truncation disclosure (the #1482 convention)."""
    s = s or ""
    return s if len(s) <= n else s[:n] + " …[truncated]"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_texts(cache_path: Path, needed: set[int]) -> dict[int, dict]:
    """ci -> text row from the local #1482 cache (text-mode iteration — never
    ``splitlines()``, #950). Fail-loud with the regen recipe on any miss."""
    found: dict[int, dict] = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ci = int(row["ci"])
            if ci in needed:
                found[ci] = row
    missing = needed - set(found)
    if missing:
        raise RuntimeError(
            f"{len(missing)} needed cis missing from text cache {cache_path} "
            f"(e.g. {sorted(missing)[:5]}); regenerate via "
            f"scripts/issue1482_collect_holdout_texts.py"
        )
    return found


def load_percontext(args) -> list[dict]:
    """percontext_ranks.csv rows (pod P1 output; git-committed)."""
    path = FC.out_eval_dir(args) / "percontext_ranks.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels_1738(args) -> dict:
    doc = json.loads(FC.repo_banked_path(args.labels_1738).read_text())
    return doc["labels"]


def resolve_ci_fields(args) -> dict[str, dict]:
    """ci_fields.json (P0.5 export): local override, local derived copy, else
    the ~300 KB HF fetch — never the 0.46 GB manifest (plan §4)."""
    if args.ci_fields:
        return json.loads(Path(args.ci_fields).read_text())["fields"]
    local = FC._derived(args) / "ci_fields.json"
    if local.exists():
        return json.loads(local.read_text())["fields"]
    target = (
        Path("data") / "issue_2202" / ("ci_fields_smoke.json" if args.smoke else "ci_fields.json")
    )
    target = PROJECT_ROOT / target
    hub.stage_hub_file(
        FC.C.HF_DATA_REPO,
        f"{FC.hf_prefix(args)}/analysis_tensors/ci_fields.json",
        target,
        repo_type="dataset",
    )
    return json.loads(target.read_text())["fields"]


def cell_of(ci: int, fields: dict[str, dict], labels: dict) -> tuple[str, str, str]:
    """(depth_band, corpus, language) matching cell — the plan's matched-control
    + stratification axes."""
    f = fields[str(int(ci))]
    lang = labels.get(str(int(ci)), {}).get("language", "unlabeled")
    return (f["depth_band"], f["corpus"], lang)


def stratified_pick(cis: list[int], cell_fn, n_take: int, seed: int) -> list[int]:
    """Largest-remainder proportional stratified pick (the
    ``issue1738_characterize.phase_kresample_subsample`` allocation shape)."""
    strata: dict[tuple, list[int]] = {}
    for ci in cis:
        strata.setdefault(cell_fn(ci), []).append(ci)
    n_take = min(n_take, len(cis))
    keys = sorted(strata.keys())
    total = len(cis)
    quota = {k: n_take * len(strata[k]) / total for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rem = n_take - sum(alloc.values())
    for k in sorted(keys, key=lambda k: quota[k] - int(quota[k]), reverse=True)[:rem]:
        alloc[k] += 1
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for k in keys:
        pool = np.asarray(sorted(strata[k]), dtype=np.int64)
        take = min(alloc[k], len(pool))
        picked.extend(int(x) for x in rng.choice(pool, size=take, replace=False))
    return sorted(picked)


def build_population(rows: list[dict], fields: dict, labels: dict, seed: int) -> dict:
    """FAIL-1 set + cell-matched non-failure control (equalize-down fallback) +
    the sample rows. Fail ∩ control = ∅ by construction; realized per-cell
    counts reported (plan §4 Equalize-down)."""
    fail_cis = [int(r["ci"]) for r in rows if r["fail_raw_euclidean"] == "1"]
    nonfail_cis = [int(r["ci"]) for r in rows if r["fail_raw_euclidean"] == "0"]
    sample_cis = [int(r["ci"]) for r in rows if r["in_sample500"] == "1"]
    by_cell_fail: dict[tuple, list[int]] = {}
    for ci in fail_cis:
        by_cell_fail.setdefault(cell_of(ci, fields, labels), []).append(ci)
    by_cell_non: dict[tuple, list[int]] = {}
    for ci in nonfail_cis:
        by_cell_non.setdefault(cell_of(ci, fields, labels), []).append(ci)
    rng = np.random.default_rng(seed)
    control: list[int] = []
    fail_eq: list[int] = []
    per_cell: dict[str, dict] = {}
    for cell, f_cis in sorted(by_cell_fail.items()):
        avail = sorted(by_cell_non.get(cell, []))
        n_f, n_a = len(f_cis), len(avail)
        n_c = min(n_f, n_a)
        ctrl = (
            sorted(int(x) for x in rng.choice(np.asarray(avail), size=n_c, replace=False))
            if n_c
            else []
        )
        if n_a < n_f:
            # equalize-down: the FAILURE side matches the available controls in
            # this cell (for the contrast battery); all fails still get judged.
            feq = (
                sorted(
                    int(x) for x in rng.choice(np.asarray(sorted(f_cis)), size=n_c, replace=False)
                )
                if n_c
                else []
            )
        else:
            feq = sorted(f_cis)
        control.extend(ctrl)
        fail_eq.extend(feq)
        per_cell[" | ".join(cell)] = {"n_fail": n_f, "n_control": n_c, "n_fail_equalized": len(feq)}
    return {
        "fail_cis": sorted(fail_cis),
        "control_cis": sorted(control),
        "fail_eq_cis": sorted(fail_eq),
        "sample_cis": sorted(sample_cis),
        "per_cell": per_cell,
        "seed": seed,
    }


# ── P3a: population + digests ─────────────────────────────────────────────────────


def judge_dir(args) -> Path:
    d = FC.out_eval_dir(args) / "judge_labels_2202"
    d.mkdir(parents=True, exist_ok=True)
    return d


def digest_dir(args) -> Path:
    d = PROJECT_ROOT / "data" / "issue_2202" / ("digests_smoke" if args.smoke else "digests")
    d.mkdir(parents=True, exist_ok=True)
    return d


def phase_fable_digest(args) -> None:
    """Build population.json + the two Fable digest JSONLs; upload digests."""
    logger.info("[phase=p3_fable_digest] start (smoke=%s)", args.smoke)
    rows = load_percontext(args)
    fields = resolve_ci_fields(args)
    labels = load_labels_1738(args)
    pop = build_population(rows, fields, labels, FC.SEED)

    fc_doc = json.loads((FC.out_eval_dir(args) / "failures_confusion.json").read_text())
    sl_doc = json.loads((FC.out_eval_dir(args) / "sample500_lists.json").read_text())
    geom_by_ci = {int(r["ci"]): r for r in fc_doc["rows"]}
    rank_by_ci = {int(r["ci"]): float(r["rank_raw_euclidean"]) for r in rows}

    # digest 1: WORST-N by rank + stratified pick of the REMAINING failures
    n_worst = 10 if args.smoke else DIGEST_WORST
    n_strat = 10 if args.smoke else DIGEST_STRAT
    fails_sorted = sorted(pop["fail_cis"], key=lambda c: (-rank_by_ci[c], c))
    worst = fails_sorted[:n_worst]
    remaining = [c for c in fails_sorted[n_worst:]]
    strat = stratified_pick(remaining, lambda c: cell_of(c, fields, labels), n_strat, FC.SEED)
    digest1_cis = worst + strat
    sample_cis = [int(r["ci"]) for r in sl_doc["rows"]]

    conf_cis = {
        int(cf["ci"]) for ci in digest1_cis for cf in geom_by_ci.get(ci, {}).get("confusers", [])
    }
    needed = set(digest1_cis) | set(sample_cis) | conf_cis
    cache = Path(args.text_cache)
    texts = load_texts(cache, needed)

    def _excerpt(ci: int) -> dict:
        t = texts[ci]
        return {
            "history_tail": cap_text(t["history_tail"], CAP_HISTORY),
            "last_user": cap_text(t["last_user"], CAP_LAST_USER),
            "response": cap_text(t["response"], CAP_RESPONSE),
            "corpus": t.get("corpus", "?"),
        }

    d1_rows = []
    for ci in digest1_cis:
        g = geom_by_ci.get(ci, {})
        d1_rows.append(
            {
                "ci": ci,
                "rank": rank_by_ci[ci],
                "n_outrank": g.get("n_outrank"),
                "attribution": g.get("attribution", "UNKNOWN"),
                "labels_1738": labels.get(str(ci)),
                "worst_tail": ci in set(worst),
                "text": _excerpt(ci),
                "confusers": [
                    {
                        "ci": int(cf["ci"]),
                        "sims": cf.get("sims"),
                        "text": {
                            "last_user": cap_text(texts[int(cf["ci"])]["last_user"], CAP_CONFUSER),
                            "response": cap_text(texts[int(cf["ci"])]["response"], CAP_CONFUSER),
                        },
                    }
                    for cf in g.get("confusers", [])[:3]
                ],
            }
        )
    d2_rows = []
    for r in sl_doc["rows"]:
        ci = int(r["ci"])
        d2_rows.append(
            {
                "ci": ci,
                "rank": r["rank"],
                "fail": r["fail"],
                "labels_1738": labels.get(str(ci)),
                "text": _excerpt(ci),
                "retrieval_top": r["retrieval"][:5],
                "collapse_top": r["collapse"][:5],
            }
        )

    ddir = digest_dir(args)
    names = FC.shard_json_rows(d1_rows, "digest_result1", ddir)
    names += FC.shard_json_rows(d2_rows, "digest_result2", ddir)

    pop["digest1_cis"] = digest1_cis
    pop["text_cache_sha256"] = sha256_file(cache)
    pop["meta"] = FC.meta_block({"smoke": bool(args.smoke)})
    FC.atomic_json(judge_dir(args) / "population.json", pop)

    if not args.no_upload:
        dest = f"{FC.hf_prefix(args)}/digests"
        url = hub._upload_folder_filtered(
            ddir,
            repo_id=FC.C.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            allow_patterns=names,
            expected_repo_paths=[f"{dest}/{nm}" for nm in names],
        )
        if not url:
            raise RuntimeError(f"digest upload to {dest} returned no URL")
    logger.info(
        "[p3a] population fail=%d control=%d sample=%d digest1=%d",
        len(pop["fail_cis"]),
        len(pop["control_cis"]),
        len(pop["sample_cis"]),
        len(digest1_cis),
    )


# ── P3b: Fable synthesis ──────────────────────────────────────────────────────────


def fable_reply_ok(parsed: object) -> bool:
    """``response_valid`` predicate for Fable calls (three-way contract):

    1. A record with ``stop_reason == "refusal"`` is VALID-at-dispatch: the
       provider's safety classifier declined (zero content blocks,
       output_tokens≈3 at production shape) — a ~deterministic CONTENT outcome,
       so retrying it (5 attempts x 2 orgs pre-fix) is pure waste. The record
       is cached and returned; :func:`harvest_fable_results` routes it to the
       first-class ``refused`` collection (never coerced, never silent).
    2. A blank / whitespace-only reply WITHOUT the refusal stop_reason is NEVER
       a success (the fable-digest-rerun fail-fast rule): retried as
       transport-class, ``error=True`` on exhaustion, never WRITTEN to the
       cache as success (api_dispatch #1470).
    3. A CACHED non-error record whose stored result fails this predicate reads
       as a MISS — which heals the poisoned pre-fix cache (old records stored
       the reply as a plain str under ``parse_response=lambda t: t``, so they
       fail the dict check here even when non-empty). Existing cached VALID
       dict replies still pass (the refusal branch only WIDENS acceptance;
       cache keys are untouched)."""
    if not isinstance(parsed, dict):
        return False
    if parsed.get("stop_reason") == "refusal":
        return True
    return bool(str(parsed.get("text") or "").strip())


def harvest_fable_results(
    items: list[tuple[str, str]], results: dict, max_tokens: int
) -> tuple[dict, dict]:
    """Post-dispatch validation: ``(ok, refused)`` — each {id: {"text",
    "stop_reason"}} — or raise.

    A record with ``stop_reason == "refusal"`` routes to the returned
    ``refused`` collection, NEVER ``bad``: a content refusal is a
    ~deterministic first-class outcome the caller handles (per-row fallback in
    :func:`phase_fable_read`; half-split fallback at consolidation stage 1 in
    :func:`consolidate_stage1`; designed rc-25 halt at the pilot and the
    stage-2 final merge), not an error to retry (llm-judging rule 28's third
    drop class — dropped and reported, never coerced).

    HARD errors (never absorbed): a dispatch error, a blank/whitespace-only or
    wrong-shaped NON-refusal result (belt-and-suspenders with
    :func:`fable_reply_ok`), and ``stop_reason == "max_tokens"`` — a truncated
    reply would silently degrade the mode list, the exact class this repair
    targets (llm-judging rule 26). Truncation is a caller-side error rather
    than a retry: at a fixed cap it is ~deterministic, and the cache key
    fingerprints ``max_tokens``, so a cached truncated reply can never replay
    under a raised cap."""
    out: dict[str, dict] = {}
    refused: dict[str, dict] = {}
    bad = []
    for i, _p in items:
        res = results[i]
        rec = res.result if isinstance(res.result, dict) else None
        if res.error or rec is None:
            bad.append((i, res.reason or "empty_or_malformed_reply"))
        elif rec.get("stop_reason") == "refusal":
            refused[i] = rec
        elif not str(rec.get("text") or "").strip():
            bad.append((i, "empty_or_malformed_reply"))
        elif rec.get("stop_reason") == "max_tokens":
            bad.append((i, f"stop_reason=max_tokens (truncated at cap {max_tokens})"))
        else:
            out[i] = rec
    if bad:
        raise RuntimeError(f"Fable dispatch errors: {bad[:3]} ({len(bad)} total)")
    return out, refused


def fable_dispatch(
    items: list[tuple[str, str]], args, max_tokens: int = FABLE_MAX_TOKENS
) -> tuple[dict, dict]:
    """Sync dispatch of (id, prompt) items to Fable via the multi-org dispatcher
    (mandatory api_dispatch route). Returns ``(ok, refused)`` — each
    {id: {"text": str, "stop_reason": str | None}} (``refused`` =
    ``stop_reason == "refusal"`` records: cached, non-retried, first-class);
    raises on any error, blank non-refusal reply, or max_tokens truncation."""
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    ditems = [DispatchItem(item_id=i, payload=p) for i, p in items]

    def build_request(it: DispatchItem) -> dict:
        return {
            "model": FABLE_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": it.payload}],
            # Request OPTION, not a body param: an explicit timeout skips the
            # SDK's 10-min non-streaming ValueError guard, required for
            # max_tokens > ~21,333 (see FABLE_REQUEST_TIMEOUT_S). Valid on the
            # sync path only — force_path="sync" below pins it; strip this key
            # before ever routing these params to the batch path.
            "timeout": FABLE_REQUEST_TIMEOUT_S,
        }

    results = asyncio.run(
        dispatch_calls(
            ditems,
            model=FABLE_MODEL,
            build_request=build_request,
            # meta parser is PREFERRED at the parse site (#2021): persists the
            # response's stop_reason so rule-26 truncation gating has a field
            # to fire on (the prior run discarded it — fable-digest-rerun).
            parse_response=lambda t: {"text": t, "stop_reason": None},
            parse_response_meta=lambda t, sr: {"text": t, "stop_reason": sr},
            response_valid=fable_reply_ok,
            cache_dir=PROJECT_ROOT / "data" / "issue_2202" / "fable_cache",
            checkpoint_dir=PROJECT_ROOT / "data" / "issue_2202" / "fable_ckpt",
            force_path="sync",
        )
    )
    return harvest_fable_results(items, results, max_tokens)


def sanitize_mode_name(name: str) -> str:
    """snake_case, [a-z0-9_], <= 40 chars, never colliding with 'reasoning'."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:40] or "mode"
    return s + "_" if s == "reasoning" else s


def parse_modes(text: str) -> list[dict] | None:
    """Parse a Fable JSON reply into [{name, description, decision_rule}].

    Returns ``None`` when the reply does not parse to the ``{"modes": [...]}``
    schema at all (a parse FAILURE — hard-error at the caller, never silently
    absorbed as zero proposals: the fable-digest-rerun fail-fast rule), vs
    ``[]`` for a schema-valid reply whose modes list is genuinely empty (a
    legitimate "no modes found", surfaced as a warning by the caller)."""
    parsed = parse_judge_json(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("modes"), list):
        return None
    out = []
    for m in parsed["modes"]:
        if isinstance(m, dict) and m.get("name") and m.get("decision_rule"):
            out.append(
                {
                    "name": sanitize_mode_name(m["name"]),
                    "description": str(m.get("description", ""))[:300],
                    "decision_rule": str(m["decision_rule"])[:300],
                }
            )
    return out


FABLE_TASK_R1 = (
    "You are analyzing RETRIEVAL FAILURES of a linear (ridge) map from conversation-context "
    "activations to answer activations in a language model (Qwen-2.5-7B-Instruct, layer 19). "
    "For each failed case below, the map's predicted answer vector ranked at least one OTHER "
    "held-out conversation's answer above the true one. Each case shows the conversation "
    "excerpt, the true answer excerpt, per-case metadata (rank of the true answer among 9,941 "
    "candidates, attribution flag from a resampling control), and top confuser conversations "
    "with similarity numbers (cos_raw/cos_cent/cos_whiten per relation: cc=context-context, "
    "aa=answer-answer, ac=answer-confuserContext, pa=prediction-confuserAnswer).\n\n"
    "Propose CANDIDATE FAILURE MODES: recurring, countable kinds of contexts/answers the map "
    "fails on. Each mode needs a one-line DECISION RULE that a later annotator can apply "
    "yes/no to a SINGLE exchange (history + final user message + answer) WITHOUT seeing the "
    "confusers or any numbers. Output ONLY JSON: "
    '{"modes": [{"name": "snake_case_name", "description": "...", "decision_rule": "..."}]}'
)
FABLE_TASK_R2 = (
    "You are analyzing what a linear context-to-answer activation map is GOOD and BAD at. "
    "Below is a random sample of held-out conversations with the map's retrieval rank of the "
    "true answer (rank 1 = success), plus each case's nearest retrieved answers and nearest "
    "OTHER predictions. Propose modes describing what the map is good at and bad at, each "
    "with a one-line yes/no DECISION RULE applicable to a single exchange without the lists. "
    'Output ONLY JSON: {"modes": [{"name": "snake_case_name", "description": "...", '
    '"decision_rule": "..."}]}'
)


FABLE_TASK_BY_STEM = {"digest_result1": FABLE_TASK_R1, "digest_result2": FABLE_TASK_R2}


def load_digest_rows(args, stem: str) -> list[dict]:
    """Digest rows for one stem, manifest shard order — shared by the chunk
    assembly and the per-row refusal fallback (same rows, same order)."""
    ddir = digest_dir(args)
    man = json.loads((ddir / f"{stem}.manifest.json").read_text())
    rows: list[dict] = []
    for shard in man["shards"]:
        with open(ddir / shard, encoding="utf-8") as f:
            rows.extend(json.loads(ln) for ln in f if ln.strip())
    return rows


def build_fable_items(args) -> list[tuple[str, str]]:
    """Assemble the chunked (id, prompt) Fable items from the digest shards.

    Module-level (not inline in the phase) so the production-shape pilot gate
    and the smoke's chunk-assembly leg exercise the exact production chunking."""
    items: list[tuple[str, str]] = []
    for stem, task in (("digest_result1", FABLE_TASK_R1), ("digest_result2", FABLE_TASK_R2)):
        rows = load_digest_rows(args, stem)
        per = max(1, args.fable_chunk_rows)
        for k in range(0, len(rows), per):
            blk = rows[k : k + per]
            body = "\n\n---\n\n".join(json.dumps(r, ensure_ascii=False) for r in blk)
            items.append((f"{stem}_c{k // per:02d}", f"{task}\n\nCASES:\n\n{body}"))
    assert items, "no digest rows found — run --phase fable-digest first"
    return items


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """``digest_result1_c07`` -> ("digest_result1", 7). Raises on a foreign id."""
    stem, sep, k = chunk_id.rpartition("_c")
    if not sep or stem not in FABLE_TASK_BY_STEM:
        raise ValueError(f"not a digest chunk id: {chunk_id!r}")
    return stem, int(k)


def build_fable_row_items(args, chunk_ids: list[str]) -> tuple[list[tuple[str, str]], dict]:
    """Single-row fallback items for content-refused chunks.

    Returns ``(items, info)``. ``items`` are ``(row_id, prompt)`` pairs — the
    SAME per-stem task text as the chunk with a CASES block of exactly ONE row
    (no ``---`` separators); ``row_id`` = ``<chunk_id>_r<row_index>`` (e.g.
    ``digest_result1_c00_r07``), with ``row_index`` the row's GLOBAL index in
    the stem's digest rows (manifest shard order — unambiguous across chunks).
    ``info[row_id] = {"chunk", "row_index", "ci"}`` feeds the
    refusal-exclusion record."""
    per = max(1, args.fable_chunk_rows)
    rows_by_stem: dict[str, list[dict]] = {}
    items: list[tuple[str, str]] = []
    info: dict[str, dict] = {}
    for chunk_id in chunk_ids:
        stem, k = parse_chunk_id(chunk_id)
        if stem not in rows_by_stem:
            rows_by_stem[stem] = load_digest_rows(args, stem)
        rows = rows_by_stem[stem]
        for gidx in range(k * per, min((k + 1) * per, len(rows))):
            row = rows[gidx]
            rid = f"{chunk_id}_r{gidx:02d}"
            body = json.dumps(row, ensure_ascii=False)
            items.append((rid, f"{FABLE_TASK_BY_STEM[stem]}\n\nCASES:\n\n{body}"))
            info[rid] = {"chunk": chunk_id, "row_index": gidx, "ci": row.get("ci")}
    return items, info


def fable_pilot_gate(items: list[tuple[str, str]], args, out_fable: Path) -> None:
    """Rule-26 pilot at PRODUCTION shape (replaces the retired ``"Reply with
    the single word OK."`` A17 probe, which exercised auth/routing only and was
    structurally incapable of surfacing an output-budget exhaustion).

    Dispatches ONE real digest chunk — the largest-bytes one, the max-stress
    production prompt — at the production ``FABLE_MAX_TOKENS`` and gates on:
    (a) a non-empty text block and (b) ``stop_reason != "max_tokens"`` (both
    enforced as hard errors inside :func:`fable_dispatch`), plus (c) the reply
    parses to the ``{"modes": [...]}`` schema (an EMPTY modes list from a
    schema-valid reply passes with a warning — a genuine "no modes found" is
    acceptable at pilot; the gate's core is (a)+(b)). Designed halt: rc 25.
    The pilot reply is cached under the production cache key, so the full
    dispatch re-serves it from cache — zero double-spend."""
    pilot_id, pilot_prompt = max(items, key=lambda ip: len(ip[1]))
    logger.info(
        "[p3b] pilot gate: chunk %s (%d chars, max_tokens=%d)",
        pilot_id,
        len(pilot_prompt),
        FABLE_MAX_TOKENS,
    )
    try:
        ok, refused = fable_dispatch([(pilot_id, pilot_prompt)], args)
    except Exception as exc:  # designed halt: report written, distinct rc
        FC.atomic_json(out_fable / "probe.json", {"ok": False, "error": str(exc)[:500]})
        logger.error("[p3b] Fable pilot gate FAILED (empty/truncated/error): %s", exc)
        sys.exit(RC_FABLE)
    if pilot_id in refused:  # a refusing pilot still halts rc 25 by design
        FC.atomic_json(
            out_fable / "probe.json",
            {"ok": False, "error": "pilot chunk refused (stop_reason=refusal)"},
        )
        logger.error("[p3b] Fable pilot gate FAILED: pilot chunk %s refused", pilot_id)
        sys.exit(RC_FABLE)
    rec = ok[pilot_id]
    modes = parse_modes(rec["text"])
    report = {
        "ok": modes is not None,
        "pilot_chunk": pilot_id,
        "prompt_chars": len(pilot_prompt),
        "reply_chars": len(rec["text"]),
        "stop_reason": rec["stop_reason"],
        "n_modes": None if modes is None else len(modes),
        "max_tokens": FABLE_MAX_TOKENS,
    }
    FC.atomic_json(out_fable / "probe.json", report)
    if modes is None:
        logger.error(
            "[p3b] pilot gate FAILED: reply (%d chars, stop_reason=%s) does not parse "
            "to the {'modes': [...]} schema",
            len(rec["text"]),
            rec["stop_reason"],
        )
        sys.exit(RC_FABLE)
    if not modes:
        logger.warning("[p3b] pilot chunk parsed to ZERO modes (schema-valid) — proceeding")
    logger.info("[p3b] pilot gate PASS (%d modes, stop_reason=%s)", len(modes), rec["stop_reason"])


# The consolidation instruction (wording UNCHANGED from the retired
# single-call implementation) — shared verbatim by every hierarchical stage:
# stage-1 batches, half-split retries, and the stage-2 final merge.
FABLE_CONSOL_INSTRUCTION = (
    "You proposed candidate failure/success modes for a context-to-answer activation map "
    "across several chunks. Consolidate them into AT MOST "
    f"{MODE_CAP} canonical modes: merge near-duplicates, drop modes that cannot be decided "
    "yes/no from a single exchange (history + final user message + answer) alone, keep the "
    "most countable. Each decision_rule must be ONE line, self-contained, and applicable "
    "without seeing confusers or numbers. Output ONLY JSON: "
    '{"modes": [{"name": "snake_case_name", "description": "...", "decision_rule": "...", '
    '"result": 1 or 2}]}\n\nPROPOSED MODES:\n\n'
)


def consolidation_batches(pool: list[dict]) -> list[list[dict]]:
    """Deterministic contiguous stage-1 batches of ``FABLE_CONSOL_BATCH``
    (read as a module global at call time so tests can shrink the constant)."""
    per = max(1, FABLE_CONSOL_BATCH)
    return [pool[k : k + per] for k in range(0, len(pool), per)]


def consolidation_prompt(pool: list[dict]) -> str:
    """The consolidation instruction over one pool of proposal/mode dicts."""
    return FABLE_CONSOL_INSTRUCTION + json.dumps(pool, ensure_ascii=False, indent=1)


def consolidate_stage1(
    pool: list[dict], args, out_fable: Path, id_prefix: str = "consolidation_b"
) -> tuple[list[dict], list[dict], int]:
    """ONE stage-1 batch-consolidation pass (hierarchical consolidation).

    Splits ``pool`` into deterministic contiguous batches
    (:func:`consolidation_batches`), dispatches ALL batches in ONE
    ``fable_dispatch`` call (ids ``<id_prefix>00``, ``01``, ... — parallel +
    cached), then applies the half-split retry to content-refused batches: the
    refused batch is split in half ONCE (first half gets the odd proposal) and
    both halves re-dispatched together in a second single call; a half that
    STILL refuses has its proposals DROPPED and reported (never coerced, never
    silent). Any NON-refusal reply that fails the ``{"modes": [...]}`` schema
    is a HARD rc-25 halt — a schema failure is a bug, not content (the
    fable-digest-rerun fail-fast rule). Every reply persists verbatim to
    ``out_fable/<id>.json`` (audit).

    Returns ``(modes, dropped, n_batches)``: the concatenated surviving modes
    (each stamped with its ``source_batch`` id, capped at MODE_CAP per reply),
    the drop records (``{"batch", "half", "n_proposals"}``), and the batch
    count."""
    batches = consolidation_batches(pool)
    items = [(f"{id_prefix}{j:02d}", consolidation_prompt(b)) for j, b in enumerate(batches)]
    logger.info(
        "[p3b] consolidation stage 1: %d proposals -> %d batches (batch_size=%d, id_prefix=%s)",
        len(pool),
        len(batches),
        FABLE_CONSOL_BATCH,
        id_prefix,
    )

    def _persist(i: str, rec: dict) -> None:
        FC.atomic_json(
            out_fable / f"{i}.json",
            {"id": i, "raw": rec["text"], "stop_reason": rec["stop_reason"]},
        )

    def _parse_or_halt(i: str, rec: dict) -> list[dict]:
        modes = parse_modes(rec["text"])
        if modes is None:  # non-refusal unparseable: hard halt, never absorbed
            logger.error(
                "[p3b] consolidation batch %s: reply (%d chars, stop_reason=%s) does not "
                "parse to the {'modes': [...]} schema",
                i,
                len(rec["text"]),
                rec["stop_reason"],
            )
            sys.exit(RC_FABLE)
        if not modes:
            logger.warning("[p3b] consolidation batch %s: schema-valid reply but ZERO modes", i)
        return modes[:MODE_CAP]

    ok, refused = fable_dispatch(items, args)
    out_modes: list[dict] = []
    dropped: list[dict] = []
    half_items: list[tuple[str, str]] = []
    half_info: dict[str, tuple[str, int, list[dict]]] = {}
    for j, (bid, _prompt) in enumerate(items):
        if bid in ok:
            _persist(bid, ok[bid])
            for m in _parse_or_halt(bid, ok[bid]):
                m["source_batch"] = bid
                out_modes.append(m)
            continue
        rec = refused[bid]  # content-refused batch: half-split ONCE
        _persist(bid, rec)
        blk = batches[j]
        mid = (len(blk) + 1) // 2
        for h, half in enumerate((blk[:mid], blk[mid:])):
            if not half:  # a 1-proposal batch has an empty second half
                continue
            hid = f"{bid}_h{h}"
            half_items.append((hid, consolidation_prompt(half)))
            half_info[hid] = (bid, h, half)
    if half_items:
        h_ok, h_ref = fable_dispatch(half_items, args)
        for hid, _p in half_items:
            bid, h, half = half_info[hid]
            if hid in h_ref:  # still refusing -> proposals dropped-and-reported
                _persist(hid, h_ref[hid])
                dropped.append({"batch": bid, "half": h, "n_proposals": len(half)})
                continue
            _persist(hid, h_ok[hid])
            for m in _parse_or_halt(hid, h_ok[hid]):
                m["source_batch"] = hid
                out_modes.append(m)
    return out_modes, dropped, len(batches)


def phase_fable_read(args) -> None:
    """P3b — production-shape pilot gate, chunked synthesis (with a per-row
    fallback for content-refused chunks: refused rows are dropped-and-reported,
    never coerced, never silent), hierarchical consolidation → modes.json."""
    logger.info("[phase=p3_fable_read] start")
    out_fable = FC.out_eval_dir(args) / "fable_reads"
    out_fable.mkdir(parents=True, exist_ok=True)

    items = build_fable_items(args)
    fable_pilot_gate(items, args, out_fable)

    replies, refused = fable_dispatch(items, args)
    proposals: list[dict] = []
    zero_mode: list[str] = []
    unparseable: list[str] = []

    def _collect(i: str, rec: dict) -> None:
        """Persist one raw reply verbatim + fold its parsed modes into the pool."""
        FC.atomic_json(
            out_fable / f"{i}.json",
            {"id": i, "raw": rec["text"], "stop_reason": rec["stop_reason"]},
        )
        chunk_modes = parse_modes(rec["text"])
        if chunk_modes is None:
            unparseable.append(i)
            return
        if not chunk_modes:
            zero_mode.append(i)
            logger.warning(
                "[p3b] %s: schema-valid reply but ZERO modes proposed (%d chars, stop_reason=%s)",
                i,
                len(rec["text"]),
                rec["stop_reason"],
            )
        for m in chunk_modes:
            m["source_chunk"] = i
            proposals.append(m)

    for i, _ in items:
        if i in replies:
            _collect(i, replies[i])
        else:  # refused chunk: persist the raw refusal record verbatim (audit)
            rec = refused[i]
            FC.atomic_json(
                out_fable / f"{i}.json",
                {"id": i, "raw": rec["text"], "stop_reason": rec["stop_reason"]},
            )

    # Per-row fallback: a chunk refusal is content-keyed, so single-row
    # re-dispatch (same task text, CASES block of exactly one row) isolates the
    # triggering rows; the rest of the chunk still contributes its modes. Rows
    # that individually refuse are EXCLUDED — recorded, never coerced.
    fallback_chunks = sorted(refused)
    exclusion_entries: list[dict] = []
    if fallback_chunks:
        row_items, row_info = build_fable_row_items(args, fallback_chunks)
        row_replies, row_refused = fable_dispatch(row_items, args)
        for rid, _ in row_items:
            entry = dict(row_info[rid])
            if rid in row_refused:
                entry["stage"] = "row-refused"  # individually refused -> dropped
                rec = row_refused[rid]
                FC.atomic_json(
                    out_fable / f"{rid}.json",
                    {"id": rid, "raw": rec["text"], "stop_reason": rec["stop_reason"]},
                )
            else:
                entry["stage"] = "chunk-fallback"  # recovered via per-row fallback
                _collect(rid, row_replies[rid])
            exclusion_entries.append(entry)

    n_rows_dropped = sum(1 for e in exclusion_entries if e["stage"] == "row-refused")
    refusal_meta = {
        "n_rows_dropped": n_rows_dropped,
        "n_chunks_fallback": len(fallback_chunks),
    }
    excl_doc = {
        **refusal_meta,
        "fallback_chunks": fallback_chunks,
        # One entry per row of every refused chunk: stage "row-refused" =
        # the single-row fallback ALSO refused -> row EXCLUDED from the
        # mode pool; stage "chunk-fallback" = re-dispatched per-row and
        # recovered (contributes modes as normal).
        "entries": exclusion_entries,
        # Consolidation-stage drops (hierarchical consolidation) are filled in
        # AFTER stage 1 below; this early write keeps the row-exclusion record
        # crash-safe across a later consolidation rc-25 halt.
        "consolidation_dropped": [],
        "meta": FC.meta_block(),
    }
    FC.atomic_json(out_fable / "refusal_exclusions.json", excl_doc)
    logger.info(
        "[p3b] refusal exclusions: %d rows dropped (%d chunks fell back per-row)",
        n_rows_dropped,
        len(fallback_chunks),
    )

    if unparseable:
        logger.error(
            "[p3b] %d/%d chunk replies do not parse to the {'modes': [...]} schema: %s",
            len(unparseable),
            len(items),
            unparseable,
        )
        sys.exit(RC_FABLE)

    # Hierarchical consolidation (fable-digest-rerun crash-fix 2): the single
    # all-proposals call refuses at aggregate size (see FABLE_CONSOL_BATCH),
    # so a pool over the batch size runs a stage-1 batch pass (half-split
    # retry, still-refusing halves dropped-and-reported) before the stage-2
    # final merge; a pool at or under the batch size keeps the single final
    # call (the fast path — prior behavior).
    pool = proposals
    consol_dropped: list[dict] = []
    n_batches = 1  # fast path: the whole pool rides the single final call
    stages = 1
    if len(pool) > FABLE_CONSOL_BATCH:
        pool, consol_dropped, n_batches = consolidate_stage1(pool, args, out_fable)
        stages = 2
        if len(pool) > FABLE_CONSOL_STAGE2_MAX:  # ONE conditional extra pass max
            pool, extra_dropped, extra_batches = consolidate_stage1(
                pool, args, out_fable, id_prefix="consolidation_x"
            )
            consol_dropped += extra_dropped
            n_batches += extra_batches
            stages = 3
    n_consol_dropped = sum(d["n_proposals"] for d in consol_dropped)
    consol_meta = {
        "n_proposals": len(proposals),
        "n_batches": n_batches,
        "batch_size": FABLE_CONSOL_BATCH,
        "n_proposals_dropped": n_consol_dropped,
        "stages": stages,
    }
    # ALWAYS emitted (0-count included) — a fix-engaged signal of this round.
    logger.info(
        "[p3b] consolidation exclusions: %d proposals dropped (%d batch-halves refused)",
        n_consol_dropped,
        len(consol_dropped),
    )
    excl_doc["consolidation_dropped"] = consol_dropped
    FC.atomic_json(out_fable / "refusal_exclusions.json", excl_doc)

    if stages > 1:
        logger.info("[p3b] consolidation stage 2: final merge over %d stage-1 modes", len(pool))
    consol_ok, consol_ref = fable_dispatch([("consolidation", consolidation_prompt(pool))], args)
    # A refused stage-2/final consolidation ({"text": "", "stop_reason":
    # "refusal"}) folds into the same designed halt as a schema parse failure:
    # empty text parses to None -> uniq == [] -> rc 25 ("consolidation
    # unparseable/empty").
    consol = (
        consol_ok["consolidation"] if "consolidation" in consol_ok else consol_ref["consolidation"]
    )
    FC.atomic_json(
        out_fable / "consolidation.json",
        {"raw": consol["text"], "stop_reason": consol["stop_reason"]},
    )
    # None (schema parse failure) folds into the existing empty-modes designed
    # halt below (rc 25 — "consolidation unparseable/empty").
    modes = (parse_modes(consol["text"]) or [])[:MODE_CAP]
    # de-duplicate sanitized names deterministically
    seen: set[str] = set()
    uniq = []
    for m in modes:
        nm = m["name"]
        k = 2
        while nm in seen:
            nm = f"{m['name']}_{k}"
            k += 1
        m["name"] = nm
        seen.add(nm)
        uniq.append(m)
    if not uniq:
        FC.atomic_json(
            out_fable / "modes.json",
            {
                "modes": [],
                "note": "consolidation unparseable/empty",
                "refusal_exclusions": refusal_meta,
                "consolidation": consol_meta,
                "meta": FC.meta_block(),
            },
        )
        logger.error("[p3b] Fable consolidation yielded no parseable modes")
        sys.exit(RC_FABLE)
    FC.atomic_json(
        out_fable / "modes.json",
        {
            "modes": uniq,
            "n_proposals": len(proposals),
            # Coverage caveat for the analyzer: rows whose content never
            # contributed mode proposals (content-refused at row grain), plus
            # the hierarchical-consolidation topology + its dropped-proposal
            # count (proposals lost to still-refusing batch-halves).
            "refusal_exclusions": refusal_meta,
            "consolidation": consol_meta,
            "meta": FC.meta_block(),
        },
    )
    logger.info("[p3b] %d canonical modes from %d proposals", len(uniq), len(proposals))


# ── P4: Sonnet re-labeling wave ───────────────────────────────────────────────────


def load_modes(args) -> list[dict]:
    return json.loads((FC.out_eval_dir(args) / "fable_reads" / "modes.json").read_text())["modes"]


def rubric_system(modes: list[dict]) -> str:
    """Multi-field yes/no rubric over the Fable-named modes (reason-then-score;
    llm-judging rules 6/7/9)."""
    lines = [
        "You are a careful data annotator labeling real user-assistant chat exchanges "
        "against a fixed list of candidate failure modes of a representation map. You never "
        "refuse to CATEGORIZE content. Judge each mode from the exchange alone using its "
        "decision rule. First reason briefly, then output ONLY a JSON object with EXACTLY "
        "these keys:",
        '  "reasoning": one to three sentences.',
    ]
    for m in modes:
        lines.append(f'  "{m["name"]}": "yes" or "no" — {m["decision_rule"]}')
    return "\n".join(lines)


def rubric_user_msg(text_row: dict) -> str:
    """Judge-visible excerpt at the locked caps (arm-SYMMETRIC: the exchange
    only — no confuser text, so fail/control/sample arms see one instrument)."""
    return (
        f"Corpus: {text_row.get('corpus', '?')}\n\n"
        f"=== CONVERSATION HISTORY (tail, truncated to {CAP_HISTORY} chars) ===\n"
        f"{text_row['history_tail'][-CAP_HISTORY:]}\n\n"
        f"=== FINAL USER MESSAGE (truncated to {CAP_LAST_USER} chars) ===\n"
        f"{text_row['last_user'][:CAP_LAST_USER]}\n\n"
        f"=== ASSISTANT ANSWER (truncated to {CAP_RESPONSE} chars) ===\n"
        f"{text_row['response'][:CAP_RESPONSE]}\n\n"
        "Label every mode field per the system instructions. Reason briefly, then output "
        "the JSON object."
    )


def wave_items(args) -> tuple[list[tuple[str, str, str, str]], dict[str, str]]:
    """(custom_id, question, completion, user_msg) items + {custom_id: arm}."""
    pop = json.loads((judge_dir(args) / "population.json").read_text())
    arms = (
        [(f"f{c}", "fail", c) for c in pop["fail_cis"]]
        + [(f"c{c}", "control", c) for c in pop["control_cis"]]
        + [(f"s{c}", "sample", c) for c in pop["sample_cis"]]
    )
    texts = load_texts(Path(args.text_cache), {c for _, _, c in arms})
    items = []
    arm_of: dict[str, str] = {}
    for cid, arm, ci in arms:
        t = texts[ci]
        items.append(
            (cid, t["last_user"][:CAP_LAST_USER], t["response"][:CAP_RESPONSE], rubric_user_msg(t))
        )
        arm_of[cid] = arm
    return items, arm_of


def validate_mode_label(parsed: object, modes: list[dict]) -> dict | None:
    """Schema-validate one judge return (yes/no per mode); None = content drop
    (drop-never-coerce, rule 9)."""
    if not isinstance(parsed, dict):
        return None
    out = {}
    for m in modes:
        v = parsed.get(m["name"])
        if not isinstance(v, str):
            return None
        v = v.strip().lower()
        if v not in ("yes", "no"):
            return None
        out[m["name"]] = v
    return out


def run_judge(args, tag: str, items: list, system: str, force_batch: bool) -> dict:
    """dispatch_judge_items with a per-tag checkpoint dir (fresh pilot/retest
    dirs — rule 26 cache discipline / the parent's dispatch_{tag} shape)."""
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items, keep_raw_judge_text

    with keep_raw_judge_text():
        return dispatch_judge_items(
            items,
            judge_model=JUDGE_MODEL,
            judge_system_prompt=system,
            max_tokens=JUDGE_MAX_TOKENS,
            threshold_base=1 if force_batch else 2000,
            checkpoint_dir=judge_dir(args) / f"dispatch_{tag}",
            error_dict_factory=lambda reason: {"error": True, "reason": reason},
        )


def tally_results(results: dict, modes: list[dict], arm_of: dict) -> dict:
    """Labels + per-arm drop split (content vs transport vs other) + stop_reason
    tally over one dispatch's results."""
    labels: dict[str, dict] = {}
    drops: dict[str, dict] = {}
    stop_tally: dict[str, int] = {}
    raw_rows: list[dict] = []
    for cid, res in results.items():
        arm = arm_of.get(cid.removeprefix("rt_"), "?")
        d = drops.setdefault(arm, {"content": 0, "transport_loss": 0, "error_other": 0, "n": 0})
        d["n"] += 1
        if isinstance(res, dict) and res.get("error"):
            d["transport_loss" if is_transport_error_dict(res) else "error_other"] += 1
            raw_rows.append(
                {"custom_id": cid, "error": True, "reason": str(res.get("reason"))[:300]}
            )
            continue
        sr = res.get("stop_reason") if isinstance(res, dict) else None
        if isinstance(sr, str):
            stop_tally[sr] = stop_tally.get(sr, 0) + 1
        raw_rows.append(
            {
                "custom_id": cid,
                "stop_reason": sr,
                "raw": (res or {}).get("_raw_text", "") if isinstance(res, dict) else "",
            }
        )
        lab = validate_mode_label(res, modes)
        if lab is None:
            d["content"] += 1
            continue
        labels[cid] = lab
    return {"labels": labels, "drops": drops, "stop_reason_tally": stop_tally, "raw_rows": raw_rows}


def write_raw(args, tag: str, raw_rows: list[dict]) -> None:
    rdir = judge_dir(args) / "raw"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{tag}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in raw_rows) + "\n", encoding="utf-8"
    )


def pilot_gate(tally: dict) -> dict:
    """Rule-26 gate: zero max_tokens stops AND per-arm parse-fail < 2%."""
    n_trunc = tally["stop_reason_tally"].get("max_tokens", 0)
    per_arm = {}
    ok = n_trunc == 0
    for arm, d in tally["drops"].items():
        denom = max(1, d["n"] - d["transport_loss"])
        rate = d["content"] / denom
        per_arm[arm] = {"parse_fail_rate": rate, **d}
        if rate >= 0.02:
            ok = False
    return {
        "verdict": "PASS" if ok else "FAIL",
        "n_max_tokens_stops": n_trunc,
        "per_arm": per_arm,
        "stop_reason_tally": tally["stop_reason_tally"],
        "max_tokens": JUDGE_MAX_TOKENS,
    }


def phase_sonnet_pilot(args) -> None:
    """P4a — live 5-request forced-batch probe + the pilot gate (rc 22 on FAIL)."""
    logger.info("[phase=p4_sonnet_pilot] start")
    modes = load_modes(args)
    system = rubric_system(modes)
    items, arm_of = wave_items(args)
    rng = np.random.default_rng(FC.SEED)

    # live forced-batch smoke (~5 requests) through the SAME builder (#763)
    smoke_idx = rng.choice(len(items), size=min(5, len(items)), replace=False)
    smoke_res = run_judge(args, "smoke5", [items[i] for i in smoke_idx], system, force_batch=True)
    inval = [
        (cid, r.get("reason"))
        for cid, r in smoke_res.items()
        if isinstance(r, dict) and r.get("error") and "invalid_request" in str(r.get("reason"))
    ]
    if inval:
        FC.atomic_json(
            judge_dir(args) / "pilot_gate_report.json",
            {"verdict": "FAIL", "stage": "live-smoke5", "invalid_request": inval[:5]},
        )
        logger.error("[p4a] live forced-batch smoke got invalid_request errors: %s", inval[:2])
        sys.exit(RC_PILOT)

    n_pilot = 20 if args.smoke else min(PILOT_N, len(items))
    pilot_idx = rng.choice(len(items), size=n_pilot, replace=False)
    res = run_judge(args, "pilot", [items[i] for i in pilot_idx], system, force_batch=True)
    tly = tally_results(res, modes, arm_of)
    write_raw(args, "pilot", tly["raw_rows"])
    report = pilot_gate(tly)
    report["n_pilot"] = n_pilot
    report["meta"] = FC.meta_block({"smoke": bool(args.smoke)})
    FC.atomic_json(judge_dir(args) / "pilot_gate_report.json", report)
    if report["verdict"] != "PASS":
        logger.error("[p4a] PILOT GATE FAILED: %s", json.dumps(report["per_arm"])[:400])
        sys.exit(RC_PILOT)
    logger.info("[p4a] pilot gate PASS (n=%d)", n_pilot)


def phase_sonnet_wave(args) -> None:
    """P4b — the full production wave (Batch API route at n ~ 4.2k)."""
    logger.info("[phase=p4_sonnet_wave] start")
    gate = json.loads((judge_dir(args) / "pilot_gate_report.json").read_text())
    assert gate["verdict"] == "PASS", "pilot gate has not PASSed — refusing the production wave"
    modes = load_modes(args)
    system = rubric_system(modes)
    items, arm_of = wave_items(args)
    if args.smoke:
        rng = np.random.default_rng(FC.SEED + 1)
        keep = set(rng.choice(len(items), size=min(20, len(items)), replace=False).tolist())
        items = [it for k, it in enumerate(items) if k in keep]
    res = run_judge(args, "main", items, system, force_batch=False)
    tly = tally_results(res, modes, arm_of)
    write_raw(args, "main", tly["raw_rows"])
    FC.atomic_json(
        judge_dir(args) / "labels_main.json",
        {
            "labels": tly["labels"],
            "arms": {cid: arm_of[cid] for cid, _q, _a, _u in items},
            "drops": tly["drops"],
            "stop_reason_tally": tly["stop_reason_tally"],
            "modes": modes,
            "meta": FC.meta_block({"smoke": bool(args.smoke)}),
        },
    )
    logger.info("[p4b] wave labeled %d/%d", len(tly["labels"]), len(items))


def phase_retest(args) -> None:
    """P4c — rt_-prefixed test-retest, κ per mode, demotion, final labels.json."""
    logger.info("[phase=p4_retest] start")
    main_doc = json.loads((judge_dir(args) / "labels_main.json").read_text())
    modes = main_doc["modes"]
    system = rubric_system(modes)
    items, arm_of = wave_items(args)
    judged = [it for it in items if it[0] in main_doc["labels"]]
    n_rt = min(20 if args.smoke else RETEST_N, len(judged))
    rng = np.random.default_rng(FC.SEED + 2)
    pick = rng.choice(len(judged), size=n_rt, replace=False)
    rt_items = [(f"rt_{judged[i][0]}", *judged[i][1:]) for i in pick]
    res = run_judge(args, "retest", rt_items, system, force_batch=False)
    tly = tally_results(res, modes, arm_of)
    write_raw(args, "retest", tly["raw_rows"])

    kappa: dict[str, dict] = {}
    demoted: list[str] = []
    for m in modes:
        a, b = [], []
        for i in pick:
            cid = judged[i][0]
            l1 = main_doc["labels"].get(cid)
            l2 = tly["labels"].get(f"rt_{cid}")
            if l1 and l2:
                a.append(l1[m["name"]])
                b.append(l2[m["name"]])
        kap = A82._cohens_kappa(a, b)
        kept = bool(np.isfinite(kap) and kap >= KAPPA_DEMOTE)
        kappa[m["name"]] = {
            "n": len(a),
            "kappa": None if not np.isfinite(kap) else float(kap),
            "kept": kept,
        }
        if not kept:
            demoted.append(m["name"])

    doc = {
        "n_items": len(items),
        "n_labeled": len(main_doc["labels"]),
        "labels": main_doc["labels"],
        "arms": main_doc["arms"],
        "drops": main_doc["drops"],
        "stop_reason_tally": main_doc["stop_reason_tally"],
        "retest_drops": tly["drops"],
        "modes": modes,
        "demoted_modes": demoted,
        "test_retest_kappa": kappa,
        "kappa_demotion_threshold": KAPPA_DEMOTE,
        "judge_model": JUDGE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": "API default",
        "n_draws": 1,
        "rubric_sha256_system": hashlib.sha256(system.encode()).hexdigest()[:16],
        "excerpt_caps": {
            "last_user": CAP_LAST_USER,
            "history_tail": CAP_HISTORY,
            "response": CAP_RESPONSE,
        },
        "meta": FC.meta_block({"smoke": bool(args.smoke)}),
    }
    FC.atomic_json(judge_dir(args) / "labels.json", doc)
    upload_judge_mirror(args)
    logger.info(
        "[p4c] retest n=%d; demoted modes: %s", n_rt, ", ".join(demoted) if demoted else "(none)"
    )


def upload_judge_mirror(args) -> None:
    """HF mirror of the judge dir (labels/population/pilot report/raw) — the
    plan's issue2202_ctxfail/judge/ destination (text/JSON: uploads always)."""
    if args.no_upload:
        logger.info("[upload] judge mirror SKIPPED (--no-upload)")
        return
    jdir = judge_dir(args)
    rel = [
        str(p.relative_to(jdir))
        for p in jdir.rglob("*")
        if p.is_file() and (p.suffix in (".json", ".jsonl")) and "dispatch_" not in str(p)
    ]
    dest = f"{FC.hf_prefix(args)}/judge"
    url = hub._upload_folder_filtered(
        jdir,
        repo_id=FC.C.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=dest,
        allow_patterns=rel,
        expected_repo_paths=[f"{dest}/{r}" for r in rel],
    )
    if not url:
        raise RuntimeError(f"judge mirror upload to {dest} returned no URL")


# ── registry + main ───────────────────────────────────────────────────────────────

PHASES = {
    "fable-digest": phase_fable_digest,
    "fable-read": phase_fable_read,
    "sonnet-pilot": phase_sonnet_pilot,
    "sonnet-wave": phase_sonnet_wave,
    "retest": phase_retest,
}
PHASE_ORDER = ["fable-digest", "fable-read", "sonnet-pilot", "sonnet-wave", "retest"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2202 Fable synthesis + Sonnet label wave")
    ap.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--import-check", action="store_true", dest="import_check")
    ap.add_argument("--list-phases", action="store_true", dest="list_phases")
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_2202"))
    ap.add_argument("--hf-prefix", default=FC.HF_PREFIX_2202)
    ap.add_argument("--text-cache", default=DEFAULT_TEXT_CACHE, dest="text_cache")
    ap.add_argument("--labels-1738", default=LABELS_1738_REL, dest="labels_1738")
    ap.add_argument("--ci-fields", default="", dest="ci_fields")
    ap.add_argument(
        "--work-root",
        default="/workspace/data/issue_2202",
        help="only used to locate a pod-local derived/ci_fields.json when present",
    )
    # 25 rows/chunk (was 100): result1 rows average ~5.7 KB, so 100-row chunks
    # were ~565 KB (~140k tokens) and exhausted the Fable reasoning+output
    # budget (fable-digest-rerun). Chunk count rises; consolidation merges
    # across chunks, so the count is free.
    ap.add_argument("--fable-chunk-rows", type=int, default=25, dest="fable_chunk_rows")
    ap.add_argument("--no-upload", action="store_true", dest="no_upload")
    return ap


def _import_check() -> None:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("import-check OK: issue2202_labels")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    if args.list_phases:
        print(json.dumps(PHASE_ORDER))
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check / --list-phases)")
    args.work_root = Path(args.work_root)
    for ph in PHASE_ORDER if args.phase == "all" else [args.phase]:
        PHASES[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
