"""P3 judge driver for issue #2091 — VM-side, Anthropic Batch API (plan §4.2 P3).

Judges every GREEDY (k=1, temperature=0.0) completion the unit-A pod run
produced, under the SAME per-rung instrument as the banked #1739 K=5 wave:

- ``sycophancy`` / ``evil`` / all-WildChat rows: the 0-100 graded per-trait
  ``eval_prompt`` rubric (``judging.load_trait_rubric`` semantics) via
  ``judging.judge_items_graded`` (rubric-keyed cache, drop-never-coerce,
  transport-vs-content split — llm-judging.md rules 9/22/24).
- ``hallucination`` OWN-RUNG rows (hal_train / hal_nqopen / hal_simpleqa —
  rows carrying ``answer_aliases``): alias-match FIRST
  (``judging.split_hallucination_items``), then ``judging.HALLU_ABSTAIN_RUBRIC``
  judged over ONLY the non-alias-correct rollouts, then
  ``judging.three_way_classify`` -> ``dv_build.build_three_way_dv``.
- WildChat rows are judged under ALL THREE trait rubrics (the wcrung
  convention); the hallucination WILDCHAT leg is the graded trait rubric
  (``hallucination_trait`` wave), NOT the alias path (no aliases there).

W1 PRE-SPEND rubric parity smoke (before ANY judge call; ZERO API calls):

1. Resolve all rubrics deterministically — the local
   ``data/issue_779/artifacts/<trait>.json`` cache (the exact bytes
   ``issue779_common.load_extraction_artifacts`` serves) + the git-resident
   ``EVIL_ARTIFACTS`` constant + the git-resident ``HALLU_ABSTAIN_RUBRIC``.
   The Sonnet-regeneration fallback of ``load_e1_assets`` is NEVER taken (it
   would mint a NEW instrument and break parity with the banked DVs) — this
   driver reads the cache file directly and FAILS LOUD on a miss.
2. Byte-match against NAMED banked realized-instrument references:
   (a) ``judge_system_prompt_sha256`` from #1739's api_dispatch batch
   checkpoints (``.dispatch/dispatch_*/state.json``) vs the sha256 of the
   graded_judge system prompt this driver would dispatch — a mismatch ABORTS;
   (b) a JudgeCache KEY-REPRODUCTION probe: recompute the rule-22 cache key
   (``EPM_JUDGE_CACHE_KEY_V2`` + ``rubric_fingerprint`` over the CURRENT
   resolved rubric) for banked #1739 judged items and check the
   ``{16-hex}.json`` files exist in the banked cache dirs — >=1 hit proves the
   full realized instrument (model + system + user template) is byte-identical
   at sha level; 0 hits over >= MIN_PROBE items ABORTS (re-source the banked
   realized text verbatim; never Sonnet-regenerate).
3. A wave with NO locatable banked reference (``hallucination_trait`` — the
   wcrung trait caches are not VM-resident) DEGRADES EXPLICITLY: deterministic
   re-resolution + rubric sha256/fingerprints recorded in
   ``rubric_parity.json`` + a carried instrument caveat on the greedy DV
   column — never a silent pass.

Pilot gate first (rule 26): ``eval.judge_pilot.judge_pilot_gate`` per rubric
wave (~200 draws total across all rubrics x arms) at the EXACT production
instrument, fresh pilot cache dirs; a gate FAIL exits rc=7 (a DESIGNED
artifact-routed halt, never an anonymous crash — gotchas.md pilot-gate entry).
Two (wave, arm) cells carry a recorded PARSE-FAIL waiver
(``PILOT_WAIVE_PARSE_FAIL_ARMS`` — rule 26(b)'s explained-content-drop
escape; analysis: epm:progress v35 on #2091); truncation and the
effective-draws floor stay unwaivable, and every other arm still gates.

Outputs: ``eval_results/issue_2091/greedy_dv/<behavior>.json`` in the
labeling.json row schema (``build_labeling_dv`` / ``build_three_way_dv``).
Bulky save_raw judge outputs land under ``data/issue_2091/judge_raw/`` (the
upload-policy free-text-size rule; ``--phase upload-raw`` pushes them to the
HF data repo) — small tallies + final DVs stay in git under eval_results/.

CONTENT HYGIENE: logs, tallies, and this driver's stdout carry ids, counts,
scores, and sha256 fingerprints — NEVER rubric text, prompt text, or
completion text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue2091_judge")

# ── script-mode sys.path guard (#823) ─────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_syspath() -> None:
    """Deferred ``scripts.*`` imports need the repo root on sys.path (#823)."""
    sentinel = _REPO_ROOT / "scripts" / "issue2091_stage_contexts.py"
    assert sentinel.exists(), f"repo-root sentinel missing: {sentinel}"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))


# ── pins (plan §4.2 P3 / §11) ─────────────────────────────────────────────────
SEED = 20910  # Source: plan §11 (issue seed)
HF_PREFIX = "issue2091_decode"  # Source: unit A (issue2091_stage_contexts.HF_PREFIX)
DATA_REPO = "superkaiba1/explore-persona-space-data"
BEHAVIORS = ("sycophancy", "evil", "hallucination")
N_JUDGE_DRAWS = 3  # Source: #1739 constants.N_JUDGE_DRAWS (instrument parity)
JUDGE_TEMPERATURE = 1.0  # Source: #1739 constants.JUDGE_TEMPERATURE
JUDGE_MAX_TOKENS = 1024  # Source: plan §4.2 P3 (rule-23 single-rationale floor;
#   deliberately ABOVE #1739's realized 400 — its truncation re-judge motivated
#   the 2026-08-02 floor raise; rubric_fingerprint excludes max_tokens, so
#   cache identity with the banked instrument is unaffected)
JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # project judge pin (CLAUDE.md)
K_ROLLOUTS_GREEDY = 1  # Source: unit A pod pin (greedy single rollout)
MIN_PROBE_ITEMS = 10  # cache-key probe: 0 hits over >= this many judged items => mismatch
N_PROBE_ITEMS = 20

DEFAULT_OUT_ROOT = _REPO_ROOT / "eval_results" / "issue_2091" / "greedy_dv"
DEFAULT_RAW_ROOT = _REPO_ROOT / "data" / "issue_2091" / "judge_raw"
DEFAULT_CACHE_ROOT = _REPO_ROOT / "data" / "issue_2091" / "judge_cache"
DEFAULT_ROLLOUT_ROOT = _REPO_ROOT / "data" / "issue_2091" / "hf_dl"  # stage_hub_prefix MIRROR root
DEFAULT_BANKED_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue1739_judge")

WAVE_NAMES = ("sycophancy_trait", "evil_trait", "hallucination_trait", "hallucination_abstain")


@dataclass(frozen=True)
class Wave:
    """One rubric wave: a rubric + the banked #1739 reference it parity-checks against."""

    name: str
    behavior: str  # DV behavior this wave feeds
    kind: str  # "trait" | "abstain"
    # banked reference for the cache-key reproduction probe (None => no VM-resident
    # realized reference for THIS rubric => explicit degrade; plan §4.2 P3):
    banked_behavior: str | None
    nonalias_only: bool = False  # probe/judge only non-alias-correct banked items


WAVES: tuple[Wave, ...] = (
    Wave("sycophancy_trait", "sycophancy", "trait", "sycophancy"),
    Wave("evil_trait", "evil", "trait", "evil"),
    # #1739 realized the hallucination TRAIT rubric only in the wcrung wave,
    # whose judge cache is not VM-resident (verified 2026-08-05: the local
    # issue1739_wcrung dir holds only judge_wave.log) -> pre-registered
    # explicit-degrade path.
    Wave("hallucination_trait", "hallucination", "trait", None),
    Wave("hallucination_abstain", "hallucination", "abstain", "hallucination", nonalias_only=True),
)
WAVES_BY_NAME = {w.name: w for w in WAVES}

# ── G2 pilot-gate parse-fail waivers (rule 26(b) explained-content-drop escape) ─
# Root-cause analysis + decision record: epm:progress v35 on #2091. The judge
# scores CORRECTLY; ``parse_judge_json`` recovers a trailing JSON object after
# reasoning prose but NOT a trailing bare integer, so real scores are dropped
# (zero truncation, zero refusals, every draw stop_reason=end_turn). The banked
# #1739 K=5 comparison arm carries the SAME censoring class (8.30% on the same
# wildchat/hallucination cell; 28.27% on the evil trait rungs) and CANNOT be
# re-parsed (its caches store only the 13-byte parsed result), so fixing the
# parser for the greedy arm alone would split the instrument across the two
# arms of the headline Δ — the parser gap is filed separately as #2109 and is
# deliberately NOT fixed here (instrument parity with the banked arm).
# The waiver covers PARSE-FAIL ONLY (``judge_pilot_gate`` semantics): a
# truncation signature or an under-floor effective-draw count on these arms
# still FAILs, every arm not listed here still gates at the 2% bar, and an
# unknown/renamed arm name here raises ``ValueError`` at gate time (fail-loud).
PILOT_WAIVE_PARSE_FAIL_ARMS: dict[str, tuple[str, ...]] = {
    # 28.0% parse-fail at n=50 (14/50 draws dropped — a real rate, not a
    # small-n threshold artifact; epm:progress v35 on #2091).
    "hallucination_trait": ("wildchat",),
    # 6.25% = 1/16 — at 16 draws/arm the smallest observable non-zero rate IS
    # 6.25%, so the cell is underpowered against the 2% bar; the banked
    # same-cell rate is 2.28% (epm:progress v35 on #2091).
    "sycophancy_trait": ("wildchat",),
}


# ── deferred imports (resolved by _import_check; #606/#1739 discipline) ──────
def _judging():
    from explore_persona_space.experiments.issue_1739 import judging

    return judging


def _dv_build():
    from explore_persona_space.experiments.issue_1739 import dv_build

    return dv_build


def _graded_judge():
    from explore_persona_space.eval import graded_judge

    return graded_judge


def _batch_judge():
    from explore_persona_space.eval import batch_judge

    return batch_judge


def _hub():
    from explore_persona_space.orchestrate import hub

    return hub


def _rung_jobs():
    _ensure_repo_root_on_syspath()
    from scripts.issue2091_stage_contexts import RUNG_JOBS

    return RUNG_JOBS


def _git_commit() -> str:
    """Best-effort git sha (tolerant — the git-less-lane rule; VM driver, so normally present)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unavailable-no-git-checkout"
    except OSError:
        return "unavailable-no-git-checkout"


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Atomic JSON write (tmp + os.replace, same dir => same filesystem)."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


# ── rubric resolution (NEVER the Sonnet-regen fallback; plan §4.2 P3 W1) ─────
def _main_root() -> Path | None:
    """Worktree-safe main-checkout root (git-common-dir parent — gotchas.md)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip()).parent
    return None


def _artifacts_dir_candidates() -> list[Path]:
    """Trait-rubric sources, STRONGEST realized reference first (W1 re-source rule).

    1. The #1739 REALIZED persisted copies (``load_e1_assets`` writes
       ``inputs/e1_assets/<behavior>.json`` at judge time) — byte-identical to
       the instrument the banked caches were keyed under (verified: the main
       root's ``data/issue_779`` sycophancy copy MISMATCHES the banked cache
       0/20 while the issue-1739 worktree's realized copy is the dispatched
       text; probed 2026-08-05).
    2. The issue-1739 worktree's own ``data/issue_779/artifacts`` tree (the
       tree #1739's waves actually loaded from).
    3. This worktree's / the main checkout's ``data/issue_779/artifacts``
       (deterministic re-resolution — the explicit-degrade path when no banked
       reference exists; ``data/`` is gitignored so a worktree run needs the
       main-root fallback).
    """
    cands: list[Path] = []
    main_root = _main_root()
    if main_root is not None:
        wt_1739 = main_root / ".claude" / "worktrees" / "issue-1739"
        cands.append(wt_1739 / "data" / "issue_1739" / "inputs" / "e1_assets")
        cands.append(wt_1739 / "data" / "issue_779" / "artifacts")
    cands.append(_REPO_ROOT / "data" / "issue_779" / "artifacts")
    if main_root is not None:
        cands.append(main_root / "data" / "issue_779" / "artifacts")
    return cands


def resolve_rubric_no_regen(behavior: str, *, artifacts_dir: Path | None = None) -> str:
    """Resolve a trait ``eval_prompt`` WITHOUT the Sonnet-regeneration fallback.

    Mirrors ``issue779_common.load_extraction_artifacts`` content-for-content:
    ``evil`` returns the git-resident ``EVIL_ARTIFACTS`` constant; other traits
    read the untracked ``data/issue_779/artifacts/<trait>.json`` cache and FAIL
    LOUD on a miss (the regen fallback would mint a NEW instrument — banned by
    plan §4.2 P3). Asserts both ``{question}``/``{answer}`` slots are present
    (rule 27(b) presence check).
    """
    if behavior == "evil":
        _ensure_repo_root_on_syspath()
        from scripts.issue779_common import EVIL_ARTIFACTS

        eval_prompt = EVIL_ARTIFACTS["eval_prompt"]
    else:
        dirs = [artifacts_dir] if artifacts_dir else _artifacts_dir_candidates()
        cache = next(
            (d / f"{behavior}.json" for d in dirs if d and (d / f"{behavior}.json").is_file()), None
        )
        if cache is None:
            raise FileNotFoundError(
                f"trait-artifact cache for {behavior!r} not found in {[str(d) for d in dirs]}; "
                "REFUSING the Sonnet-regeneration fallback (instrument parity, plan §4.2 P3)"
            )
        assets = json.loads(cache.read_text())
        eval_prompt = assets["eval_prompt"]
    assert isinstance(eval_prompt, str) and eval_prompt.strip(), behavior
    for slot in ("{question}", "{answer}"):
        assert slot in eval_prompt, f"{behavior} rubric missing substitution slot {slot}"
    return eval_prompt


def resolve_all_rubrics(*, artifacts_dir: Path | None = None) -> dict[str, str]:
    """wave name -> rubric text (handled in code only; report sha256 fingerprints)."""
    out: dict[str, str] = {}
    for wave in WAVES:
        if wave.kind == "trait":
            out[wave.name] = resolve_rubric_no_regen(wave.behavior, artifacts_dir=artifacts_dir)
        else:
            rubric = _judging().HALLU_ABSTAIN_RUBRIC
            for slot in ("{question}", "{answer}"):
                assert slot in rubric, f"abstain rubric missing slot {slot}"
            out[wave.name] = rubric
    return out


def rubric_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def wave_rubric_fingerprint(eval_prompt: str, *, judge_model: str = JUDGE_MODEL) -> str:
    """The rule-22 rubric identity fp, via the SAME construction judge_graded uses."""
    gj = _graded_judge()
    bj = _batch_judge()
    system_prompt, _ = gj._rubric_system_and_user(eval_prompt)

    def format_user_msg(question: str, answer: str) -> str:
        return eval_prompt.replace("{question}", question).replace("{answer}", answer)

    return bj.rubric_fingerprint(judge_model, system_prompt, format_user_msg)


def judge_system_sha256(eval_prompt: str) -> str:
    """sha256 of the graded_judge SYSTEM prompt this driver would dispatch."""
    gj = _graded_judge()
    system_prompt, _ = gj._rubric_system_and_user(eval_prompt)
    return hashlib.sha256(system_prompt.encode()).hexdigest()


# ── W1 parity smoke ───────────────────────────────────────────────────────────
def _banked_state_files(banked_root: Path) -> list[Path]:
    return sorted(banked_root.glob("results/judge/*/*cache_*/.dispatch/dispatch_*/state.json"))


def _banked_cache_dirs(banked_root: Path, behavior: str) -> list[Path]:
    dirs = sorted(banked_root.glob(f"results/judge/{behavior}/*cache_*"))
    return [d for d in dirs if d.is_dir() and any(d.glob("*.json"))]


def _banked_shard_files(banked_root: Path, behavior: str) -> list[Path]:
    return sorted(
        banked_root.glob(
            f"shards/issue1739_ctxmap/raw_completions/labeling_{behavior}.shard*.jsonl"
        )
    )


def _iter_banked_payloads(shard_files: list[Path], *, limit: int):
    """Yield banked labeling rollout payloads (text handled in code, never printed).

    The #1739 shards are PACKED (`{"src", "doc"}`, one line per source file);
    non-rollout docs (`_manifest.json` meta sidecars) lack `query`/`completion`
    and are skipped.
    """
    n = 0
    for shard in shard_files:
        with shard.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)["doc"]
                if not isinstance(doc, dict) or "query" not in doc or "completion" not in doc:
                    continue  # packed meta/manifest sidecar, not a rollout payload
                yield doc
                n += 1
                if n >= limit:
                    return


def cache_key_reproduction_probe(
    wave: Wave, eval_prompt: str, *, banked_root: Path, n_probe: int = N_PROBE_ITEMS
) -> dict:
    """Reproduce rule-22 cache keys for banked judged items against the banked cache.

    >=1 hit  => the FULL realized instrument (judge model + system prompt + user
                template) is byte-identical at sha level ("verified").
    0 hits over >= MIN_PROBE_ITEMS probed judged items => "mismatch" (ABORT).
    Banked cache dirs / shards unlocatable => "unrecoverable" (explicit degrade).
    """
    if wave.banked_behavior is None:
        return {
            "status": "unrecoverable",
            "reason": "no VM-resident realized reference for this rubric",
        }
    cache_dirs = _banked_cache_dirs(banked_root, wave.banked_behavior)
    shard_files = _banked_shard_files(banked_root, wave.banked_behavior)
    if not cache_dirs or not shard_files:
        return {
            "status": "unrecoverable",
            "reason": f"banked cache dirs ({len(cache_dirs)}) or shards ({len(shard_files)}) not found",
        }
    judging = _judging()
    bj = _batch_judge()
    rubric_key = wave_rubric_fingerprint(eval_prompt)
    n_probed = 0
    n_hits = 0
    # Over-scan the shard stream: the nonalias filter discards alias-correct rows.
    for payload in _iter_banked_payloads(shard_files, limit=n_probe * 40):
        if wave.nonalias_only:
            aliases = payload.get("answer_aliases") or []
            if not aliases or judging.alias_correct(payload["completion"], aliases):
                continue
        key = bj.JudgeCache._hash_key(
            payload["query"], payload["completion"], rubric_key=rubric_key
        )
        n_probed += 1
        if any((d / f"{key}.json").exists() for d in cache_dirs):
            n_hits += 1
        if n_probed >= n_probe:
            break
    if n_hits >= 1:
        status = "verified"
    elif n_probed >= MIN_PROBE_ITEMS:
        status = "mismatch"
    else:
        status = "unrecoverable"
    return {
        "status": status,
        "n_probed": n_probed,
        "n_hits": n_hits,
        "rubric_key": rubric_key,
        "cache_dirs": [str(d) for d in cache_dirs],
        "n_shard_files": len(shard_files),
    }


def run_rubric_parity_smoke(args: argparse.Namespace) -> dict:
    """W1 pre-spend parity smoke (ZERO API calls). Persists rubric_parity.json."""
    banked_root = Path(args.banked_root)
    rubrics = resolve_all_rubrics(artifacts_dir=args.rubric_artifacts_dir)

    # (a) system-prompt sha byte-match vs the banked dispatch checkpoints.
    current_system_sha = judge_system_sha256(next(iter(rubrics.values())))
    state_files = _banked_state_files(banked_root)
    banked_system_shas = {}
    for sf in state_files:
        try:
            st = json.loads(sf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sha = st.get("judge_system_prompt_sha256")
        if sha:
            banked_system_shas[str(sf)] = sha
    system_status = "unrecoverable"
    if banked_system_shas:
        system_status = (
            "verified"
            if all(sha == current_system_sha for sha in banked_system_shas.values())
            else "mismatch"
        )

    # (b) per-wave cache-key reproduction probes.
    per_wave: dict[str, dict] = {}
    for wave in WAVES:
        probe = cache_key_reproduction_probe(wave, rubrics[wave.name], banked_root=banked_root)
        probe["rubric_sha256"] = rubric_sha256(rubrics[wave.name])
        probe["rubric_fingerprint"] = wave_rubric_fingerprint(rubrics[wave.name])
        per_wave[wave.name] = probe
        logger.info(
            "[parity] wave=%s status=%s n_probed=%s n_hits=%s sha=%s fp=%s",
            wave.name,
            probe["status"],
            probe.get("n_probed"),
            probe.get("n_hits"),
            probe["rubric_sha256"][:16],
            probe["rubric_fingerprint"],
        )

    statuses = [system_status] + [p["status"] for p in per_wave.values()]
    if "mismatch" in statuses:
        overall = "abort"
    elif "unrecoverable" in statuses:
        overall = "degraded"
    else:
        overall = "verified"
    report = {
        "overall": overall,
        "system_prompt": {
            "current_sha256": current_system_sha,
            "status": system_status,
            "banked_state_files": banked_system_shas,
        },
        "waves": per_wave,
        "instrument_caveat": (
            None
            if overall == "verified"
            else "fresh-resolved vs banked-realized instrument not byte-verified for the "
            "non-'verified' waves above; carried onto the greedy DV column (plan §4.2 P3)"
        ),
        "judge_model": JUDGE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "n_draws": N_JUDGE_DRAWS,
        "temperature": JUDGE_TEMPERATURE,
        "banked_root": str(banked_root),
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out_root) / "rubric_parity.json"
    _write_json_atomic(out, report)
    # Persist the RESOLVED instrument texts so the run is self-contained: the
    # strongest source (the #1739 worktree's realized e1_assets copies) is
    # janitor-reapable, and units C/D + the HF inputs upload need the realized
    # text, not just fingerprints (instrument content stays in files, never in
    # logs/reports — trigger-dense discipline).
    for wave in WAVES:
        _write_json_atomic(
            Path(args.out_root) / "resolved_rubrics" / f"{wave.name}.json",
            {
                "wave": wave.name,
                "behavior": wave.behavior,
                "kind": wave.kind,
                "eval_prompt": rubrics[wave.name],
                "rubric_sha256": per_wave[wave.name]["rubric_sha256"],
                "rubric_fingerprint": per_wave[wave.name]["rubric_fingerprint"],
                "parity_status": per_wave[wave.name]["status"],
            },
        )
    logger.info("[parity] overall=%s -> %s", overall, out)
    if overall == "abort":
        raise SystemExit(
            "rubric parity ABORT: a recovered banked realized-instrument reference "
            "MISMATCHES the freshly-resolved rubric — re-source the banked realized text "
            f"verbatim (never Sonnet-regenerate). See {out}."
        )
    return report


# ── staging + rollout loading ─────────────────────────────────────────────────
def job_shard_dir(rollout_root: Path, job_name: str) -> Path:
    """Consumed path = <mirror root>/<full HF prefix> (stage_hub_prefix contract)."""
    return Path(rollout_root) / HF_PREFIX / "raw_completions" / "greedy" / job_name


def stage_rollouts(args: argparse.Namespace) -> dict:
    """Stage every rung-job's greedy rollout shards from HF (scoped, retried)."""
    hub = _hub()
    staged: dict[str, int] = {}
    for job in _rung_jobs():
        dest = job_shard_dir(args.rollout_root, job.name)
        existing = sorted(dest.glob("*.shard*.jsonl"))
        if existing and not args.force:
            staged[job.name] = len(existing)
            logger.info("[stage] %s: %d shards present, skipping", job.name, len(existing))
            continue
        prefix = f"{HF_PREFIX}/raw_completions/greedy/{job.name}"
        paths = hub.stage_hub_prefix(DATA_REPO, prefix, args.rollout_root, repo_type="dataset")
        staged[job.name] = len(paths)
        logger.info("[stage] %s: staged %d files", job.name, len(paths))
    return staged


def _is_rollout_doc(doc: object) -> bool:
    """Schema predicate: a packed row is a rollout iff BOTH keys are present.

    NEVER key on row index — the per-job ``_manifest.json`` lands at idx 0 of
    shard00 today, but that is positional luck a future pack-ordering change
    would silently break (#1190/#1739 pack contract: one line per SOURCE FILE,
    manifest included).
    """
    return isinstance(doc, dict) and "context_id" in doc and "rollout_k" in doc


def load_job_rollouts(rollout_root: Path, job_name: str, *, limit: int | None = None) -> list[dict]:
    """Load the greedy labeling rollout payloads for one rung-job from its shards.

    Packed-format aware: each line is ``{"src": <path relative to the raw
    root>, "doc": <original file JSON>}`` and the per-job ``_manifest.json``
    is packed as a row too. Non-rollout rows are filtered by ``_is_rollout_doc``
    (schema, never index), and on a FULL read (no ``limit`` cut) the surviving
    rollout count is VERIFIED against the manifest's ``n_kept * k_rollouts`` —
    fail-loud on mismatch (a partial/corrupted pack) or on a pack with no
    manifest row. The crashed 2026-08-06 pilot's "2001 rollouts" log line was
    the manifest row miscounted as a rollout; a silent skip would have hidden
    that class, the count check keeps it observable.
    """
    shard_dir = job_shard_dir(rollout_root, job_name)
    shards = sorted(shard_dir.glob("*.shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no rollout shards for job {job_name!r} under {shard_dir}")
    payloads: list[dict] = []
    manifests: list[dict] = []
    n_skipped_other = 0
    truncated = False
    for shard in shards:
        if truncated:
            break
        with shard.open(encoding="utf-8") as fh:  # text-mode iteration (JSONL rule)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                doc = row.get("doc")
                if not _is_rollout_doc(doc):
                    # non-rollout packed entry: the per-job manifest, or an
                    # unexpected sidecar (counted + logged, never dereferenced).
                    if str(row.get("src", "")).endswith("_manifest.json"):
                        manifests.append(doc if isinstance(doc, dict) else {})
                    else:
                        n_skipped_other += 1
                    continue
                if int(doc.get("rollout_k", 0)) >= K_ROLLOUTS_GREEDY:
                    raise ValueError(
                        f"{job_name}: rollout_k={doc.get('rollout_k')} >= K_ROLLOUTS_GREEDY"
                    )
                payloads.append(doc)
                if limit is not None and len(payloads) >= limit:
                    truncated = True
                    break
    logger.info(
        "[collect] %s: kept %d rollout rows; filtered %d non-rollout packed rows "
        "(%d manifest, %d other)%s",
        job_name,
        len(payloads),
        len(manifests) + n_skipped_other,
        len(manifests),
        n_skipped_other,
        " [limit-truncated: completeness check skipped]" if truncated else "",
    )
    if not truncated:
        if not manifests:
            raise ValueError(
                f"{job_name}: no _manifest.json row in packed shards under {shard_dir} — "
                "cannot verify rollout completeness against n_kept (malformed pack)"
            )
        expected = sum(int(m["n_kept"]) * int(m.get("k_rollouts", 1)) for m in manifests)
        if len(payloads) != expected:
            raise ValueError(
                f"{job_name}: packed rollout-row count {len(payloads)} != manifest "
                f"n_kept*k_rollouts {expected} — partial or corrupted pack under {shard_dir}"
            )
    return payloads


@dataclass
class WaveItems:
    """Collected judge inputs for one rubric wave."""

    items: list[tuple[str, str, str]]  # (item_id, question, completion)
    arm_by_item: dict[str, str]  # item_id -> rung-job name (pilot arms + drop split)
    meta_by_context: dict[str, dict]  # context_id -> {rung, split, group_key, finish_reason}
    alias_correct: dict[str, bool] | None = None  # abstain wave only: ALL own-rung items


def collect_wave_items(args: argparse.Namespace) -> dict[str, WaveItems]:
    """Route every greedy rollout to its rubric wave(s) per the RUNG_JOBS registry."""
    judging = _judging()
    waves: dict[str, WaveItems] = {
        w.name: WaveItems(items=[], arm_by_item={}, meta_by_context={}) for w in WAVES
    }
    hallu_own_rung: list[tuple[str, dict]] = []  # (job_name, payload)
    for job in _rung_jobs():
        rollouts = load_job_rollouts(args.rollout_root, job.name, limit=args.limit)
        logger.info(
            "[collect] %s: %d rollouts, judge_behaviors=%s",
            job.name,
            len(rollouts),
            list(job.judge_behaviors),
        )
        for behavior in job.judge_behaviors:
            if behavior == "hallucination" and job.gen_behavior == "hallucination":
                hallu_own_rung.extend((job.name, p) for p in rollouts)
                continue
            wave_name = f"{behavior}_trait"
            wv = waves[wave_name]
            for p in rollouts:
                item_id = judging.rollout_item_id(p["context_id"], int(p["rollout_k"]))
                if item_id in wv.arm_by_item:
                    raise ValueError(f"duplicate item id {item_id!r} in wave {wave_name}")
                wv.items.append((item_id, p["query"], p["completion"]))
                wv.arm_by_item[item_id] = job.name
                wv.meta_by_context[p["context_id"]] = {
                    "rung": p.get("rung"),
                    "split": p.get("split"),
                    "group_key": p.get("group_key"),
                    "finish_reason": p.get("finish_reason"),
                    "rung_job": job.name,
                }

    # hallucination own-rung: alias-match first; judge only the non-correct.
    payloads = [p for _, p in hallu_own_rung]
    job_by_ctx = {p["context_id"]: jn for jn, p in hallu_own_rung}
    correct, judge_items = judging.split_hallucination_items(payloads) if payloads else ({}, [])
    wv = waves["hallucination_abstain"]
    wv.items = judge_items
    wv.alias_correct = correct
    for p in payloads:
        item_id = judging.rollout_item_id(p["context_id"], int(p["rollout_k"]))
        wv.arm_by_item[item_id] = job_by_ctx[p["context_id"]]
        wv.meta_by_context[p["context_id"]] = {
            "rung": p.get("rung"),
            "split": p.get("split"),
            "group_key": p.get("group_key"),
            "finish_reason": p.get("finish_reason"),
            "rung_job": job_by_ctx[p["context_id"]],
        }
    n_corr = sum(1 for v in (correct or {}).values() if v)
    logger.info(
        "[collect] hallucination own-rung: %d rollouts, %d alias-correct, %d to judge",
        len(payloads),
        n_corr,
        len(judge_items),
    )
    return waves


# ── pilot gate (G2, rule 26) ──────────────────────────────────────────────────
def run_pilot(
    args: argparse.Namespace, waves: dict[str, WaveItems], rubrics: dict[str, str]
) -> dict:
    """One judge_pilot_gate per rubric wave at the exact production instrument."""
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    pilot_dir = Path(args.out_root) / "pilot"
    combined: dict[str, dict] = {}
    all_pass = True
    for wave in WAVES:
        wv = waves[wave.name]
        if not wv.items:
            combined[wave.name] = {"skipped": "no items"}
            continue
        arms: dict[str, list[tuple[str, str, str]]] = {}
        for item in wv.items:
            arms.setdefault(wv.arm_by_item[item[0]], []).append(item)
        # Recorded parse-fail waivers (PILOT_WAIVE_PARSE_FAIL_ARMS above) —
        # PARSE-FAIL only; an unknown arm name raises ValueError (fail-loud,
        # deliberately NOT suppressed: a typo'd/renamed arm must never
        # silently waive nothing).
        waived_arms = PILOT_WAIVE_PARSE_FAIL_ARMS.get(wave.name, ())
        report = judge_pilot_gate(
            arms,
            rubrics[wave.name],
            max_tokens=JUDGE_MAX_TOKENS,
            cache_dir=Path(args.cache_root) / "pilot" / wave.name,  # fresh pilot partition
            save_raw_dir=Path(args.raw_root) / "pilot" / wave.name,
            target_total_draws=args.pilot_draws_per_wave,
            judge_model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            waive_parse_fail_arms=waived_arms,
            report_path=pilot_dir / f"{wave.name}_gate.json",
            seed=SEED,
        )
        for arm in waived_arms:
            logger.warning(
                "[pilot] wave=%s arm=%s parse-fail %.2f%% WAIVED "
                "(waive_parse_fail_arms; recorded analysis: epm:progress v35 on #2091; "
                "parser gap filed as #2109 — truncation stays unwaivable)",
                wave.name,
                arm,
                100.0 * report.arms[arm].parse_fail_rate,
            )
        combined[wave.name] = report.to_json()
        all_pass = all_pass and report.passed
        logger.info(
            "[pilot] wave=%s verdict=%s n_draws=%d failures=%s",
            wave.name,
            report.verdict,
            report.n_total_draws,
            report.failures,
        )
    verdict = {
        "passed": all_pass,
        "waves": combined,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(pilot_dir / "gate_report.json", verdict)
    if not all_pass:
        # DESIGNED artifact-routed halt (gotchas.md pilot-gate entry): rc=7.
        logger.error("[pilot] GATE FAIL — see %s", pilot_dir / "gate_report.json")
        raise SystemExit(7)
    return verdict


# ── production judge waves ────────────────────────────────────────────────────
def _require_gates(args: argparse.Namespace) -> None:
    parity = Path(args.out_root) / "rubric_parity.json"
    if not parity.is_file():
        raise SystemExit(
            f"rubric parity report missing ({parity}) — run --phase rubric-smoke first"
        )
    overall = json.loads(parity.read_text())["overall"]
    if overall == "abort":
        raise SystemExit("rubric parity verdict is ABORT — refusing judge spend")
    gate = Path(args.out_root) / "pilot" / "gate_report.json"
    if not args.dry_run:
        if not gate.is_file():
            raise SystemExit(
                f"pilot gate report missing ({gate}) — run --phase pilot first (rule 26)"
            )
        if not json.loads(gate.read_text()).get("passed"):
            raise SystemExit("pilot gate report is FAIL — refusing production judge spend")


def run_judge(
    args: argparse.Namespace, waves: dict[str, WaveItems], rubrics: dict[str, str]
) -> dict:
    """Production Batch-API judge waves; per-wave tallies persisted the moment each lands."""
    _require_gates(args)
    judging = _judging()
    out: dict[str, dict] = {}
    for wave in WAVES:
        wv = waves[wave.name]
        tallies_path = Path(args.out_root) / "raw" / f"tallies_{wave.name}.json"
        if tallies_path.is_file() and not args.force:
            logger.info("[judge] wave=%s tallies present, skipping (resume)", wave.name)
            out[wave.name] = json.loads(tallies_path.read_text())
            continue
        if not wv.items:
            logger.info("[judge] wave=%s: no items, skipping", wave.name)
            continue
        logger.info("[judge] wave=%s: %d items x %d draws", wave.name, len(wv.items), N_JUDGE_DRAWS)
        result = judging.judge_items_graded(
            wv.items,
            rubrics[wave.name],
            cache_dir=Path(args.cache_root) / wave.name,  # per-rubric partition (rule 22 hygiene)
            save_raw=Path(args.raw_root) / f"judge_raw_{wave.name}.json",
            n_draws=N_JUDGE_DRAWS,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            judge_model=JUDGE_MODEL,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            logger.info("[judge] wave=%s dry-run complete (no API calls)", wave.name)
            continue
        tallies = judging.judge_tallies(result)
        # per-ARM content-drop vs transport-loss split (rule 24 reporting).
        per_arm: dict[str, dict[str, int]] = {}
        for item_id, n_lost in (tallies.get("per_item_transport_losses") or {}).items():
            arm = wv.arm_by_item.get(item_id, "?")
            per_arm.setdefault(arm, {"transport": 0, "content": 0})["transport"] += int(n_lost)
        for item_id, n_draws in (tallies.get("per_item_draw_counts") or {}).items():
            arm = wv.arm_by_item.get(item_id, "?")
            d = per_arm.setdefault(arm, {"transport": 0, "content": 0})
            d["content"] += max(0, N_JUDGE_DRAWS - int(n_draws))
        tallies["per_arm_drop_split"] = per_arm
        tallies["wave"] = wave.name
        tallies["n_items"] = len(wv.items)
        tallies["judge_model"] = JUDGE_MODEL
        tallies["max_tokens"] = JUDGE_MAX_TOKENS
        tallies["n_draws"] = N_JUDGE_DRAWS
        tallies["temperature"] = JUDGE_TEMPERATURE
        _write_json_atomic(tallies_path, tallies)
        if tallies.get("n_transport_lost_draws"):
            logger.warning(
                "[judge] wave=%s residual transport-loss=%d draws — re-judge before "
                "publication (rule 24(ii); surgical per-draw merge, fresh cache_dir)",
                wave.name,
                tallies["n_transport_lost_draws"],
            )
        out[wave.name] = tallies
        logger.info("[judge] wave=%s complete -> %s", wave.name, tallies_path)
    # persist the abstain wave's alias-correct map for standalone build-dv resume.
    ac = waves["hallucination_abstain"].alias_correct
    if ac is not None:
        _write_json_atomic(
            Path(args.out_root) / "raw" / "hallu_alias_correct.json", {"alias_correct": ac}
        )
    return out


# ── DV assembly (labeling.json row schema; plan §4.2 P3 outputs) ─────────────
def build_dv(args: argparse.Namespace, waves: dict[str, WaveItems]) -> dict[str, Path]:
    """Assemble eval_results/issue_2091/greedy_dv/<behavior>.json from the wave tallies."""
    dv_build = _dv_build()
    judging = _judging()
    parity = json.loads((Path(args.out_root) / "rubric_parity.json").read_text())

    def load_tallies(wave_name: str) -> dict | None:
        p = Path(args.out_root) / "raw" / f"tallies_{wave_name}.json"
        return json.loads(p.read_text()) if p.is_file() else None

    written: dict[str, Path] = {}
    common_meta = {
        "judge_model": JUDGE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "n_draws": N_JUDGE_DRAWS,
        "temperature": JUDGE_TEMPERATURE,
        "k_rollouts": K_ROLLOUTS_GREEDY,
        "rubric_parity_overall": parity["overall"],
        "instrument_caveat": parity.get("instrument_caveat"),
    }
    for behavior in ("sycophancy", "evil"):
        t = load_tallies(f"{behavior}_trait")
        if t is None:
            logger.info("[dv] %s: no tallies yet, skipping", behavior)
            continue
        wv = waves[f"{behavior}_trait"]
        rows = dv_build.build_labeling_dv(
            t["scores"],
            k_rollouts=K_ROLLOUTS_GREEDY,
            n_draws=N_JUDGE_DRAWS,
            per_item_transport_losses=t.get("per_item_transport_losses") or {},
            contexts_meta=wv.meta_by_context,
        )
        payload = {
            "behavior": behavior,
            "regime": "greedy",
            "n_contexts": len(rows),
            "n_contexts_with_dv": sum(1 for r in rows if r.get("dv") is not None),
            "rows": rows,
            "judge_meta": {
                **common_meta,
                "rubric_sha256": parity["waves"][f"{behavior}_trait"]["rubric_sha256"],
                "rubric_fingerprint": parity["waves"][f"{behavior}_trait"]["rubric_fingerprint"],
                "parity_status": parity["waves"][f"{behavior}_trait"]["status"],
                "per_arm_drop_split": t.get("per_arm_drop_split"),
                "stop_reason_tally": t.get("stop_reason_tally"),
            },
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = Path(args.out_root) / f"{behavior}.json"
        _write_json_atomic(out, payload)
        written[behavior] = out
        logger.info(
            "[dv] %s: %d rows (%d with DV) -> %s",
            behavior,
            len(rows),
            payload["n_contexts_with_dv"],
            out,
        )

    # hallucination: own-rung three-way rows + wildchat graded rows.
    t_abstain = load_tallies("hallucination_abstain")
    t_trait = load_tallies("hallucination_trait")
    ac_path = Path(args.out_root) / "raw" / "hallu_alias_correct.json"
    if t_abstain is not None and ac_path.is_file():
        alias_correct: dict[str, bool] = json.loads(ac_path.read_text())["alias_correct"]
        scores = t_abstain["scores"]
        three_way = {
            item_id: judging.three_way_classify(is_corr, None if is_corr else scores.get(item_id))
            for item_id, is_corr in alias_correct.items()
        }
        rows = dv_build.build_three_way_dv(three_way)
        wc_rows = None
        if t_trait is not None:
            wc_rows = dv_build.build_labeling_dv(
                t_trait["scores"],
                k_rollouts=K_ROLLOUTS_GREEDY,
                n_draws=N_JUDGE_DRAWS,
                per_item_transport_losses=t_trait.get("per_item_transport_losses") or {},
                contexts_meta=waves["hallucination_trait"].meta_by_context,
            )
        payload = {
            "behavior": "hallucination",
            "regime": "greedy",
            "n_contexts": len(rows),
            "n_contexts_with_dv": sum(1 for r in rows if r.get("dv") is not None),
            "rows": rows,  # own-rung three-way (dv = fabricated fraction)
            "wildchat_graded_rows": wc_rows,  # graded trait rubric over wildchat rows
            "judge_meta": {
                **common_meta,
                "abstain_rubric_sha256": parity["waves"]["hallucination_abstain"]["rubric_sha256"],
                "abstain_parity_status": parity["waves"]["hallucination_abstain"]["status"],
                "trait_rubric_sha256": parity["waves"]["hallucination_trait"]["rubric_sha256"],
                "trait_parity_status": parity["waves"]["hallucination_trait"]["status"],
                "per_arm_drop_split": t_abstain.get("per_arm_drop_split"),
                "stop_reason_tally": t_abstain.get("stop_reason_tally"),
                "fabricated_threshold": judging.HALLU_FABRICATED_THRESHOLD,
            },
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        out = Path(args.out_root) / "hallucination.json"
        _write_json_atomic(out, payload)
        written["hallucination"] = out
        logger.info(
            "[dv] hallucination: %d three-way rows, wildchat_graded=%s -> %s",
            len(rows),
            "yes" if wc_rows is not None else "pending",
            out,
        )
    return written


def upload_raw(args: argparse.Namespace) -> None:
    """Push the bulky save_raw judge outputs to the HF data repo (persist-by-default)."""
    hub = _hub()
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir() or not any(raw_root.glob("judge_raw_*.json")):
        raise FileNotFoundError(f"no judge_raw_*.json under {raw_root}")
    url = hub._upload(  # folder branch (ONE upload_folder commit; never a per-file loop)
        raw_root,
        DATA_REPO,
        "dataset",
        f"{HF_PREFIX}/judge_raw",
        raise_on_error=True,  # fail-loud (a no-path return is a tracked gap, never a warning)
    )
    if not url:
        raise RuntimeError(f"upload of {raw_root} returned no path (see hub logs)")
    logger.info("[upload-raw] %s -> %s/judge_raw", raw_root, HF_PREFIX)


# ── import-check (module-level, NOT inside main — the #1739 shadow gotcha) ───
def _import_check() -> int:
    """Resolve every deferred/lazy import this driver reaches on its real paths."""
    judging = _judging()
    dv_build = _dv_build()
    gj = _graded_judge()
    bj = _batch_judge()
    hub = _hub()
    from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
    from explore_persona_space.eval.judge_dispatch import graded_temperature  # noqa: F401
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401

    _ensure_repo_root_on_syspath()
    from scripts.issue2091_stage_contexts import RUNG_JOBS, RUNG_JOBS_BY_NAME  # noqa: F401
    from scripts.issue779_common import EVIL_ARTIFACTS, load_extraction_artifacts  # noqa: F401

    for name in (
        "rollout_item_id",
        "judge_items_graded",
        "judge_tallies",
        "split_hallucination_items",
        "three_way_classify",
        "alias_correct",
        "HALLU_ABSTAIN_RUBRIC",
        "HALLU_FABRICATED_THRESHOLD",
    ):
        assert hasattr(judging, name), name
    for name in ("build_labeling_dv", "build_three_way_dv", "parse_item_id"):
        assert hasattr(dv_build, name), name
    assert hasattr(gj, "_rubric_system_and_user")
    assert hasattr(bj, "rubric_fingerprint") and hasattr(bj.JudgeCache, "_hash_key")
    assert hasattr(hub, "stage_hub_prefix") and hasattr(hub, "_upload")
    # the #1739 function-body import-shadow class: no wave helper name may be
    # rebound inside main() (checked at compile time via co_varnames).
    for fn_name in ("run_judge", "run_pilot", "build_dv"):
        assert fn_name not in main.__code__.co_varnames, fn_name
    print("[import-check] OK: all deferred imports + symbols resolve")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=[
            "import-check",
            "rubric-smoke",
            "stage",
            "pilot",
            "judge",
            "build-dv",
            "upload-raw",
            "all",
        ],
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    ap.add_argument("--banked-root", type=Path, default=DEFAULT_BANKED_ROOT)
    ap.add_argument("--rubric-artifacts-dir", type=Path, default=None)
    ap.add_argument("--pilot-draws-per-wave", type=int, default=50)  # 4 waves x 50 ~ 200 (rule 26)
    ap.add_argument("--limit", type=int, default=None, help="per-job rollout cap (smokes only)")
    ap.add_argument(
        "--dry-run", action="store_true", help="judge phase: build requests, ZERO API calls"
    )
    ap.add_argument(
        "--force", action="store_true", help="re-stage / re-judge over existing outputs"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.phase == "import-check":
        return _import_check()
    if args.phase == "rubric-smoke":
        run_rubric_parity_smoke(args)
        return 0
    if args.phase == "stage":
        stage_rollouts(args)
        return 0
    if args.phase == "upload-raw":
        upload_raw(args)
        return 0

    rubrics = resolve_all_rubrics(artifacts_dir=args.rubric_artifacts_dir)
    if args.phase == "all":
        stage_rollouts(args)
        run_rubric_parity_smoke(args)
    collected = collect_wave_items(args)
    if args.phase in ("pilot", "all"):
        run_pilot(args, collected, rubrics)
    if args.phase in ("judge", "all"):
        run_judge(args, collected, rubrics)
    if args.phase in ("build-dv", "all"):
        build_dv(args, collected)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize (gotchas.md)
