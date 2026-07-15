#!/usr/bin/env python
"""Issue #1345 Phase 1 — generate the R3 assistant-in-narrative story corpus.

On-policy tier-2 (instruct-and-strip) generation: the story-writing
instruction lives in the system prompt (instruct) / plain-text preamble
(pretrained) and is STRIPPED before extraction — only the narrative text is
used downstream. Topic seeds = first-user-turn questions of the shared R1/R2
S-track conversations, pinned @ 7159e5804d (plan §4 Phase 1).

vLLM generation (chunked, use_tqdm=False, engine seed GEN_SEED=42,
temperature 1.0, max_new_tokens 1024), per-chunk checkpointed JSONL with a
fingerprint-gated resume. Judge filter: claude-sonnet-4-5 via api_dispatch
(reason-then-verdict, >=4 extractable Q->A turns), drop-never-coerce with a
transport re-drive. Yield floor 400/500 after one retry batch (kill
criterion, plan §7): a miss exits rc=21 so the dispatcher can halt the story
legs while the chat<->no-template phases proceed. Under --smoke the floor is
1 (any kept story proceeds) so the smoke leg ALWAYS exercises the
extract_stories phase — the production 400/500 drop-never-backfill floor is
untouched, and the rc=21 halt path stays reachable (kept=0) + unit-covered
(crash-fix r3, att-20260715-161700).

Content hygiene: seed questions are LMSYS real user text and stories are raw
model generations — this script logs COUNTS/ids only, never text.

CLI:
  uv run python scripts/issue1345_gen_stories.py --model instruct [--smoke]
  uv run python scripts/issue1345_gen_stories.py --parse-judge-only <stories.jsonl>
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402

from explore_persona_space.experiments.issue_825.common import (  # noqa: E402
    MODEL_INSTRUCT,
    MODEL_PRETRAINED,
)
from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    RESULT_RATE_LIMITED,
    RESULT_TRANSPORT,
    DispatchItem,
    dispatch_calls,
)

VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "250"))
MAX_MODEL_LEN = 4096
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - c.STORY_MAX_NEW_TOKENS - 64
# Smoke story target: the pool >= 2x target gate then needs only 6 seeds —
# strictly weaker than the production gate (2 x N_STORIES_TARGET = 1000), so
# any pool passing production passes smoke (v3 review item, test-pinned).
SMOKE_N_STORIES = 3

JUDGE_SYSTEM = (
    "You evaluate whether a short story contains extractable question-answer "
    "dialogue. A story PASSES when it contains at least "
    f"{c.STORY_MIN_TURNS} DISTINCT question->answer exchanges in which a person "
    "asks a question (quoted dialogue) and an AI assistant character named ARIA "
    "gives a substantive answer in quoted dialogue introduced by an attribution "
    'before the quote (e.g. ARIA replied: "..."). Exchanges with trivial or '
    "empty answers, unquoted dialogue, or answers not attributed to ARIA do not "
    "count. First give 2-3 sentences of reasoning, then finish with EXACTLY two "
    "final lines:\nTURNS: <integer number of qualifying exchanges>\n"
    "VERDICT: PASS or FAIL"
)


def _build_judge_request(item: DispatchItem) -> dict:
    """Messages-API params for one story-judge call (system lifted top-level)."""
    return {
        "model": c.JUDGE_MODEL,
        "max_tokens": c.JUDGE_MAX_TOKENS,
        "temperature": 0.0,
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": item.payload["story"]}],
    }


def _parse_judge_response(text: str) -> dict:
    """Extract TURNS/VERDICT from a reason-then-verdict judge reply (raise on miss)."""
    turns, verdict = None, None
    for line in text.split("\n"):
        s = line.strip()
        if s.upper().startswith("TURNS:"):
            with contextlib.suppress(ValueError, IndexError):
                turns = int(s.split(":", 1)[1].strip().split()[0])
        if s.upper().startswith("VERDICT:"):
            v = s.split(":", 1)[1].strip().upper()
            if v in ("PASS", "FAIL"):
                verdict = v
    if verdict is None:
        raise ValueError(f"judge reply missing VERDICT line ({len(text)} chars)")
    return {"verdict": verdict, "judge_turns": turns}


# ---------------------------------------------------------------------------
# Seed pool: shared R1/R2 conversation questions @ the pinned revision
# ---------------------------------------------------------------------------
def load_seed_pool(dl_dir: Path) -> list[dict]:
    """{conv_id, question} rows for the shared parent conv set (pinned rev).

    Shared set = intersection of the four parent stems' conv_ids (chat AND
    naturalistic, both models — the same subset the headline pair fits on).
    """
    id_sets = [set(c.parent_conv_ids(stem, dl_dir)) for stem in c.PARENT_STEMS]
    shared = set.intersection(*id_sets)
    track_s = c.stage_pinned_file(c.PARENT_TRACK_S_JSONL, dl_dir)
    rows = c.read_jsonl(track_s)
    pool = []
    for r in rows:
        cid = str(r.get("conv_id") or f"s{r['prompt_idx']}")
        if cid in shared:
            pool.append({"conv_id": cid, "question": r["prompt"]})
    assert pool, "seed pool empty — shared conv set did not join against track_s.jsonl"
    print(
        f"[seeds] shared convs={len(shared)} joined pool={len(pool)} "
        f"(per-stem sizes {[len(s) for s in id_sets]})",
        flush=True,
    )
    return sorted(pool, key=lambda r: r["conv_id"])


def _filter_pool_by_length(pool: list[dict], tokenizer, model_key: str) -> list[dict]:
    """Drop seeds whose FORMATTED prompt exceeds the token budget (#952 gotcha)."""
    kept, dropped = [], 0
    for row in pool:
        prompt = build_prompt(row["question"], model_key, tokenizer)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok <= PROMPT_TOKEN_BUDGET:
            kept.append(row)
        else:
            dropped += 1
    print(f"[seeds] length filter: kept={len(kept)} dropped={dropped}", flush=True)
    return kept


def build_prompt(question: str, model_key: str, tokenizer) -> str:
    """Render the tier-2 generation prompt (instruction stripped before extraction)."""
    user_msg = (
        "Write the story now. The person's questions to ARIA should be about the "
        "same topic as this question (rephrase naturally; do not copy verbatim):\n"
        f"{question}"
    )
    if model_key == "instruct":
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": c.STORY_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    # Base model: plain-text tier-2 preamble (no chat template), story follows.
    return f"{c.STORY_SYSTEM_PROMPT}\n\n{user_msg}\n\nStory:\n"


# ---------------------------------------------------------------------------
# vLLM generation (chunked + checkpointed)
# ---------------------------------------------------------------------------
def _gen_fingerprint(model_key: str, seed_ids: list[str]) -> str:
    import hashlib

    key = json.dumps(
        {
            "model": model_key,
            "gen_seed": c.GEN_SEED,
            "temperature": c.STORY_TEMPERATURE,
            "max_new_tokens": c.STORY_MAX_NEW_TOKENS,
            "system_prompt": c.STORY_SYSTEM_PROMPT,
            "seed_conv_ids": seed_ids,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def generate_stories(
    model_key: str, seeds: list[dict], out_path: Path, start_idx: int, tokenizer, llm
) -> list[dict]:
    """Generate one story per seed, checkpointing each vLLM chunk to JSONL.

    Resume: rows already in out_path with a matching fingerprint are skipped
    (fingerprint covers model, seeds, and every sampling constant).
    """
    from vllm import SamplingParams

    fp = _gen_fingerprint(model_key, [s["conv_id"] for s in seeds])
    meta_path = out_path.with_suffix(".meta.json")
    done_ids: set[str] = set()
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            done_ids = {r["story_id"] for r in c.read_jsonl(out_path)}
            print(f"[gen] resume: {len(done_ids)} stories already on disk", flush=True)
        else:
            raise RuntimeError(
                f"{out_path} exists with a DIFFERENT generation fingerprint "
                f"({meta.get('fingerprint')} != {fp}) — refusing to mix regimes; "
                "move the stale file aside"
            )
    else:
        c.write_json(meta_path, {"fingerprint": fp, "model": model_key, "n_seeds": len(seeds)})

    todo = [
        (i, s)
        for i, s in enumerate(seeds)
        if f"{model_key}_story{start_idx + i:04d}" not in done_ids
    ]
    sampling = SamplingParams(
        temperature=c.STORY_TEMPERATURE, max_tokens=c.STORY_MAX_NEW_TOKENS, seed=None
    )
    n_chunks = (len(todo) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for ci in range(0, len(todo), VLLM_CHUNK_SIZE):
        chunk = todo[ci : ci + VLLM_CHUNK_SIZE]
        prompts = [build_prompt(s["question"], model_key, tokenizer) for _, s in chunk]
        print(
            f"[vllm-chunk] gen chunk {ci // VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts, model={model_key})",
            flush=True,
        )
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        rows = []
        for (i, s), o in zip(chunk, outs, strict=True):
            rows.append(
                {
                    "story_id": f"{model_key}_story{start_idx + i:04d}",
                    "seed_conv_id": s["conv_id"],
                    "model": model_key,
                    "tier": "instruct_and_strip",
                    "story": o.outputs[0].text.strip(),
                    "finish_reason": o.outputs[0].finish_reason,
                }
            )
        c.append_jsonl(out_path, rows)
    return c.read_jsonl(out_path)


# ---------------------------------------------------------------------------
# Parse + judge filter
# ---------------------------------------------------------------------------
def parse_and_judge(
    rows: list[dict], cache_dir: Path, smoke: bool
) -> tuple[list[dict], dict, list[dict]]:
    """Parser + judge filter over generated stories; returns (kept, yield_report,
    judge_digest_rows).

    Keep = judge PASS (drop-never-coerce; malformed judge returns counted, never
    coerced; transport-class failures re-driven once, residue reported as
    transport_loss) AND parser extracts >= STORY_MIN_TURNS confident turns.
    ``judge_digest_rows`` carries one digest per story (ids/verdicts/counts only,
    NEVER story or judge text) so the judge outputs persist per Upload Policy.
    """
    parsed: dict[str, list[dict]] = {}
    for r in rows:
        parsed[r["story_id"]] = c.parse_story_turns(r["story"])

    items = [DispatchItem(item_id=r["story_id"], payload={"story": r["story"]}) for r in rows]
    results = asyncio.run(
        dispatch_calls(
            items,
            model=c.JUDGE_MODEL,
            build_request=_build_judge_request,
            parse_response=_parse_judge_response,
            cache_dir=cache_dir,
            checkpoint_dir=cache_dir / "checkpoints",
            force_path="sync" if smoke else None,
        )
    )
    redrive = [
        it
        for it in items
        if results[it.item_id].error
        and results[it.item_id].category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
    ]
    if redrive:
        print(f"[judge] re-driving {len(redrive)} transport-class failures", flush=True)
        results.update(
            asyncio.run(
                dispatch_calls(
                    redrive,
                    model=c.JUDGE_MODEL,
                    build_request=_build_judge_request,
                    parse_response=_parse_judge_response,
                    cache_dir=cache_dir,
                    checkpoint_dir=cache_dir / "checkpoints",
                    force_path="sync",
                )
            )
        )

    kept, counts = (
        [],
        {
            "n_generated": len(rows),
            "judge_pass": 0,
            "judge_fail": 0,
            "judge_malformed": 0,
            "transport_loss": 0,
            "parser_below_floor": 0,
            "kept": 0,
        },
    )
    digest_rows: list[dict] = []
    for r in rows:
        res = results[r["story_id"]]
        turns = parsed[r["story_id"]]
        confident = [
            t for t in turns if t["confidence"]["marker_exact"] or t["confidence"]["answer_len_ok"]
        ]
        digest = {
            "story_id": r["story_id"],
            "seed_conv_id": r["seed_conv_id"],
            "n_parsed_turns": len(turns),
            "n_confident_turns": len(confident),
        }
        if res.error:
            key = (
                "transport_loss"
                if res.category in (RESULT_RATE_LIMITED, RESULT_TRANSPORT)
                else "judge_malformed"
            )
            counts[key] += 1
            digest_rows.append({**digest, "judge_error_category": res.category})
            continue
        verdict = res.result["verdict"]
        judge_turns = res.result.get("judge_turns")
        digest_rows.append({**digest, "verdict": verdict, "judge_turns": judge_turns})
        counts["judge_pass" if verdict == "PASS" else "judge_fail"] += 1
        if verdict != "PASS":
            continue
        if len(confident) < c.STORY_MIN_TURNS:
            counts["parser_below_floor"] += 1
            continue
        kept.append(
            {
                **r,
                "judge_verdict": verdict,
                "judge_turns": judge_turns,
                "parsed_turns": turns,
                "n_parsed_turns": len(turns),
                "n_confident_turns": len(confident),
            }
        )
        counts["kept"] += 1
    return kept, counts, digest_rows


# ---------------------------------------------------------------------------
# Yield floor (plan §7 kill criterion; smoke floor keeps extract_stories live)
# ---------------------------------------------------------------------------
def resolve_yield_floor(smoke: bool, floor: int) -> int:
    """Yield floor for this run: 1 under --smoke, else the production floor.

    Smoke floor = 1 so ANY kept story proceeds and the smoke leg always
    exercises extract_stories (crash-fix r3: pretrained kept=1 < the old
    smoke floor 2 rc=21-halted the story regime, un-smoking the phase the
    smoke exists to cover). The production floor (STORY_YIELD_FLOOR=400/500,
    drop-never-backfill, plan §7) is untouched.
    """
    return 1 if smoke else floor


def enforce_yield_floor(n_kept: int, yield_floor: int) -> None:
    """rc=21 story-regime halt when kept < floor (plan §7 kill criterion).

    Raises SystemExit(21) — the dispatcher maps rc=21 to a story-regime halt
    (r1/r2 phases continue). Called with the smoke-resolved floor, so the
    halt path stays reachable under --smoke at kept=0.
    """
    if n_kept < yield_floor:
        print(
            f"[yield-floor] FAILED: kept={n_kept} < floor={yield_floor} — "
            "halting the story regime (plan §7 kill criterion); rc=21",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(21)


# ---------------------------------------------------------------------------
# HF persist + content-keyed resume (#1345 crash-fix r6, Upload Policy: raw
# completions persist at ALL stages — att-20260715-195605 lost 540 kept
# stories + judge outputs when the run crashed downstream of gen).
# ---------------------------------------------------------------------------
def _stories_hf_prefix(smoke: bool) -> str:
    """HF data-repo prefix for the story bundle (smoke diverts to issue1345_smoke)."""
    return f"{'issue1345_smoke' if smoke else c.HF_ISSUE_PREFIX}/raw_completions/stories"


def bundle_fingerprint(model_key: str, seed_ids_all: list[str]) -> str:
    """Content key over EVERYTHING that determines the kept-story bundle.

    Covers the generation recipe (model, engine seed, sampling constants,
    system prompt, the full ordered seed list incl. the retry reserve), the
    judge instrument (model, rubric, max_tokens), and the parser recipe
    (min-turns floor + the parser SOURCE hash — any parser change regenerates
    rather than silently reusing stale persisted stories).
    """
    import hashlib
    import inspect

    key = json.dumps(
        {
            "model": model_key,
            "gen_seed": c.GEN_SEED,
            "temperature": c.STORY_TEMPERATURE,
            "max_new_tokens": c.STORY_MAX_NEW_TOKENS,
            "system_prompt": c.STORY_SYSTEM_PROMPT,
            "seed_conv_ids": seed_ids_all,
            "judge_model": c.JUDGE_MODEL,
            "judge_system": JUDGE_SYSTEM,
            "judge_max_tokens": c.JUDGE_MAX_TOKENS,
            "story_min_turns": c.STORY_MIN_TURNS,
            "parser_source_sha": hashlib.sha256(
                inspect.getsource(c.parse_story_turns).encode()
            ).hexdigest(),
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _hf_file_exists(path_in_repo: str) -> bool:
    """Boundary wrapper: retried single-path existence probe on the data repo."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return bool(
        retry_transient(
            lambda: api.file_exists(c.HF_DATA_REPO, path_in_repo, repo_type="dataset"),
            what=f"file_exists({path_in_repo})",
        )
    )


def _hf_download_to(path_in_repo: str, dest: Path) -> Path:
    """Boundary wrapper: retried cache download + copy to ``dest`` (flat basename)."""
    import shutil

    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    p = retry_transient(
        lambda: hf_hub_download(
            c.HF_DATA_REPO,
            path_in_repo,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        ),
        what=f"hf_hub_download({path_in_repo})",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
    return dest


def _hf_upload_folder(folder: Path, path_in_repo: str, allow: list[str], msg: str) -> None:
    """Boundary wrapper: retried upload_folder (judge cache always excluded)."""
    from huggingface_hub import upload_folder

    from explore_persona_space.orchestrate.hub import retry_transient

    retry_transient(
        lambda: upload_folder(
            repo_id=c.HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            allow_patterns=allow,
            ignore_patterns=["judge_cache/*", "judge_cache/**"],
            commit_message=msg,
        ),
        what=f"upload_folder({path_in_repo})",
    )


def bundle_files(model_key: str, out_dir: Path) -> list[str]:
    """Basenames of this model's persisted story-bundle files present on disk."""
    names = [
        f"raw_stories_{model_key}.jsonl",
        f"raw_stories_{model_key}.meta.json",
        f"raw_stories_{model_key}_retry.jsonl",
        f"raw_stories_{model_key}_retry.meta.json",
        f"kept_stories_{model_key}.jsonl",
        f"story_yield_{model_key}.json",
        f"judge_results_{model_key}.jsonl",
    ]
    return [n for n in names if (out_dir / n).exists()]


def persist_story_bundle(model_key: str, out_dir: Path, fp: str, smoke: bool) -> None:
    """Upload this model's rollout text + judge outputs + manifest to HF NOW.

    Runs at gen-phase completion per model, BEFORE the yield floor can halt
    the process and BEFORE extraction starts — a downstream crash (the
    att-20260715-195605 shape) can no longer lose the kept stories. The
    manifest carries the bundle fingerprint the resume path checks.
    """
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot persist story bundle"
    files = bundle_files(model_key, out_dir)
    assert f"kept_stories_{model_key}.jsonl" in files, files
    manifest = {
        "metadata": c.metadata(c.GEN_SEED, len(files), "scripts/issue1345_gen_stories.py"),
        "model": model_key,
        "bundle_fingerprint": fp,
        "files": files,
    }
    c.write_json(out_dir / f"story_bundle_manifest_{model_key}.json", manifest)
    prefix = _stories_hf_prefix(smoke)
    _hf_upload_folder(
        out_dir,
        prefix,
        [f"*{model_key}*"],
        f"issue-1345: story bundle ({model_key}, fp {fp})",
    )
    print(
        f"[gen] persisted rollouts to {prefix} (model={model_key}, "
        f"{len(files) + 1} files, fp {fp})",
        flush=True,
    )


def try_resume_from_hf(model_key: str, fp: str, out_dir: Path, smoke: bool) -> dict | None:
    """Reuse a persisted kept-story bundle when its fingerprint matches.

    Returns the story_yield report dict on a hit (files staged into
    ``out_dir``), else None. A manifest with a DIFFERENT fingerprint is
    ignored with a log line — a recipe change never silently reuses stale
    stories (content-keyed resume, #1345 crash-fix r6).
    """
    prefix = _stories_hf_prefix(smoke)
    manifest_path = f"{prefix}/story_bundle_manifest_{model_key}.json"
    if not _hf_file_exists(manifest_path):
        return None
    local_manifest = _hf_download_to(
        manifest_path, out_dir / f"story_bundle_manifest_{model_key}.json"
    )
    manifest = json.loads(local_manifest.read_text())
    if manifest.get("bundle_fingerprint") != fp:
        print(
            f"[gen] HF story bundle for {model_key} has fingerprint "
            f"{manifest.get('bundle_fingerprint')} != {fp} — stale (recipe changed); "
            "regenerating instead of reusing",
            flush=True,
        )
        return None
    for name in manifest["files"]:
        _hf_download_to(f"{prefix}/{name}", out_dir / name)
    report = json.loads((out_dir / f"story_yield_{model_key}.json").read_text())
    print(
        f"[gen] resume-from-HF: reusing {report.get('n_kept')} persisted kept stories "
        f"(model={model_key}, fp {fp}) — generation skipped",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("instruct", "pretrained"))
    ap.add_argument("--out-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--n-stories", type=int, default=c.N_STORIES_TARGET)
    ap.add_argument("--yield-floor", type=int, default=c.STORY_YIELD_FLOOR)
    ap.add_argument("--smoke", action="store_true", help="n=3 stories, sync judge")
    ap.add_argument(
        "--parse-judge-only",
        type=Path,
        default=None,
        help="skip generation; run parser+judge on an existing stories JSONL (CPU smoke)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "judge_cache"

    if args.parse_judge_only is not None:
        rows = c.read_jsonl(args.parse_judge_only)
        kept, counts, _digest = parse_and_judge(rows, cache_dir, smoke=True)
        report = {
            "metadata": c.metadata(c.GEN_SEED, len(rows), "scripts/issue1345_gen_stories.py"),
            "mode": "parse-judge-only",
            "counts": counts,
        }
        c.write_json(out_dir / "story_yield_parsejudge.json", report)
        print(f"[done] parse-judge-only kept={len(kept)}/{len(rows)}", flush=True)
        return

    assert args.model, "--model is required unless --parse-judge-only"
    model_key = args.model
    n_target = SMOKE_N_STORIES if args.smoke else args.n_stories
    yield_floor = resolve_yield_floor(args.smoke, args.yield_floor)

    pool = load_seed_pool(args.dl_dir)

    from transformers import AutoTokenizer

    model_id = MODEL_INSTRUCT if model_key == "instruct" else MODEL_PRETRAINED
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pool = _filter_pool_by_length(pool, tokenizer, model_key)
    assert len(pool) >= 2 * n_target, f"seed pool {len(pool)} < 2x target {n_target}"

    import numpy as np

    rng = np.random.default_rng(c.GEN_SEED)
    order = rng.permutation(len(pool))
    seeds_main = [pool[i] for i in order[:n_target]]
    seeds_reserve = [pool[i] for i in order[n_target:]]

    # Content-keyed resume BEFORE any engine build: a relaunch reuses the
    # persisted kept-story bundle instead of re-generating (crash-fix r6).
    fp = bundle_fingerprint(model_key, [s["conv_id"] for s in seeds_main + seeds_reserve])
    resumed = try_resume_from_hf(model_key, fp, out_dir, args.smoke)
    if resumed is not None:
        n_kept = int(resumed["n_kept"])
        if args.smoke and n_kept < n_target:
            print(
                f"[yield-floor][smoke] shortfall: kept={n_kept}/{n_target} — proceeding "
                "(smoke floor=1 so extract_stories is exercised)",
                flush=True,
            )
        enforce_yield_floor(n_kept, yield_floor)
        print(f"[done] model={model_key} kept={n_kept}/{n_target} stories (resumed)", flush=True)
        return

    from vllm import LLM

    llm = LLM(
        model=model_id,
        seed=c.GEN_SEED,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=(
            False if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1" else None
        ),
    )

    raw_path = out_dir / f"raw_stories_{model_key}.jsonl"
    rows = generate_stories(model_key, seeds_main, raw_path, 0, tokenizer, llm)
    kept, counts, judge_digest = parse_and_judge(rows, cache_dir, args.smoke)

    retry_counts = None
    if len(kept) < n_target and seeds_reserve:
        shortfall = n_target - len(kept)
        retry_seeds = seeds_reserve[:shortfall]
        print(f"[retry] {len(kept)}/{n_target} kept — one retry batch of {len(retry_seeds)}")
        retry_path = out_dir / f"raw_stories_{model_key}_retry.jsonl"
        retry_rows = generate_stories(
            model_key, retry_seeds, retry_path, len(seeds_main), tokenizer, llm
        )
        retry_kept, retry_counts, retry_digest = parse_and_judge(retry_rows, cache_dir, args.smoke)
        kept.extend(retry_kept)
        judge_digest.extend(retry_digest)

    kept_path = out_dir / f"kept_stories_{model_key}.jsonl"
    if kept_path.exists():
        kept_path.unlink()
    c.append_jsonl(kept_path, kept)

    # Judge outputs persist alongside the rollout text (Upload Policy; digest
    # rows only — ids/verdicts/counts, never story or judge text).
    judge_path = out_dir / f"judge_results_{model_key}.jsonl"
    if judge_path.exists():
        judge_path.unlink()
    c.append_jsonl(judge_path, judge_digest)

    report = {
        "metadata": c.metadata(c.GEN_SEED, len(kept), "scripts/issue1345_gen_stories.py"),
        "model": model_key,
        "n_target": n_target,
        "yield_floor": yield_floor,
        "counts_main": counts,
        "counts_retry": retry_counts,
        "n_kept": len(kept),
        "yield_ok": len(kept) >= yield_floor,
        "answer_char_lengths": sorted(
            len(r["story"]) for r in kept[:50]
        ),  # digest only, never text
    }
    c.write_json(out_dir / f"story_yield_{model_key}.json", report)

    # Persist BEFORE the yield floor can rc=21-halt this process and before
    # extraction starts: a below-floor model's rollouts + judge outputs are
    # exactly the artifacts the Upload Policy exists to keep (crash-fix r6).
    persist_story_bundle(model_key, out_dir, fp, args.smoke)

    if args.smoke and len(kept) < n_target:
        print(
            f"[yield-floor][smoke] shortfall: kept={len(kept)}/{n_target} — proceeding "
            "(smoke floor=1 so extract_stories is exercised; production floor unchanged)",
            flush=True,
        )
    enforce_yield_floor(len(kept), yield_floor)
    print(f"[done] model={model_key} kept={len(kept)}/{n_target} stories", flush=True)


if __name__ == "__main__":
    main()
