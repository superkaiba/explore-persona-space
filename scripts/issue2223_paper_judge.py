"""Issue #2223 — project-judge replacement for the assistant-axis pipeline step 3.

The paper's `external/assistant-axis/pipeline/3_judge.py` scores role-adherence
(0-3) with OpenAI gpt-4.1-mini. This script is the ONE sanctioned deviation
(`.claude/rules/persona-vectors-recipe.md`: replace the paper's GPT judge with
the project judge, keeping the rubric intact): it reads the SAME per-role
response JSONLs (pipeline step 1 output) + the SAME per-role ``eval_prompt``
0-3 role-adherence RUBRIC (from the role JSON files), but issues the judge call
to the project judge ``claude-sonnet-4-5-20250929`` via ``api_dispatch`` (sync
below the crossover, Message Batches above it), and writes score JSONs in the
EXACT format the paper's ``4_vectors.py`` consumes:

    <output_dir>/<role>.json  == {"<label>_p<prompt_idx>_q<question_idx>": <int 0-3>}

Everything else in the paper pipeline is unchanged — steps 1,2,4,5 run verbatim
with ``--model``; only the judge CALL is swapped here.

Resumable: an existing ``<role>.json`` is loaded and already-scored keys are
skipped. ``--stub`` runs the full read/format/write/resume path on CPU with a
deterministic score and NO API call (smoke). ``--dry-run`` composes requests +
reports counts with zero API calls.

Role responses here are BENIGN character role-play (pirate, etc.) — NOT the
harmful case-study scenarios (those live in the replay phase). This script never
prints response content regardless: logs carry role names, keys, and counts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

JUDGE_MODEL = "claude-sonnet-4-5-20250929"
# The rubric asks for a bare 0-3; a project judge may add a short rationale, so
# give it headroom (llm-judging.md rule 23: a cap is not a spend).
DEFAULT_MAX_TOKENS = 512
_SCORE_RE = re.compile(r"[0-3]")


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_role_score(text: str) -> int | None:
    """Extract a 0-3 role-adherence score from judge text (mirrors the paper's parser).

    Prefer a trailing integer; else the first standalone 0-3 digit. Returns
    ``None`` on no parse (rule 9: DROP a malformed judge return, never coerce).
    """
    if not text:
        return None
    stripped = text.strip()
    # A bare trailing integer is the common well-formed shape.
    tail = re.search(r"([0-3])\s*$", stripped)
    if tail:
        return int(tail.group(1))
    m = _SCORE_RE.search(stripped)
    return int(m.group(0)) if m else None


def _load_role_eval_prompt(role_file: Path) -> str:
    return json.loads(role_file.read_text()).get("eval_prompt", "")


def _assistant_text(resp: dict) -> str:
    for msg in resp.get("conversation", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _resp_key(resp: dict) -> str:
    return f"{resp['label']}_p{resp['prompt_index']}_q{resp['question_index']}"


def _iter_responses(path: Path):
    for line in path.open(encoding="utf-8"):  # text-mode: only universal-newline splits
        line = line.strip()
        if line:
            yield json.loads(line)


def _stub_score(item_key: str) -> int:
    """Deterministic 0-3 for --stub (no API): stable hash of the key."""
    import hashlib

    return int(hashlib.sha256(item_key.encode()).hexdigest(), 16) % 4


def _collect_uncached(responses_dir: Path, roles_dir: Path, output_dir: Path, roles):
    """Enumerate (item_id -> (role, key, judge_prompt)) for every unscored response.

    item_ids are integer-indexed (charset/length-safe for the Anthropic batch
    custom_id contract, #1776) with a local map back to (role, key).
    """
    role_files = sorted(responses_dir.glob("*.jsonl"))
    if roles:
        role_files = [f for f in role_files if f.stem in roles]
    id_map: dict[str, tuple[str, str, str]] = {}
    existing: dict[str, dict[str, int]] = {}
    n = 0
    for rf in role_files:
        role = rf.stem
        out_file = output_dir / f"{role}.json"
        scored = json.loads(out_file.read_text()) if out_file.exists() else {}
        existing[role] = scored
        role_json = roles_dir / f"{role}.json"
        if not role_json.exists():
            _log(f"[judge] {role}: no role file — skip")
            continue
        template = _load_role_eval_prompt(role_json)
        if not template:
            _log(f"[judge] {role}: no eval_prompt — skip")
            continue
        for resp in _iter_responses(rf):
            key = _resp_key(resp)
            if key in scored:
                continue
            prompt = template.format(question=resp["question"], answer=_assistant_text(resp))
            id_map[f"i{n}"] = (role, key, prompt)
            n += 1
    return id_map, existing


def _write_scores(output_dir: Path, existing: dict, new_by_role: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = set(existing) | set(new_by_role)
    for role in sorted(roles):
        merged = {**existing.get(role, {}), **new_by_role.get(role, {})}
        if merged:
            (output_dir / f"{role}.json").write_text(json.dumps(merged, indent=2))


def _run_api(id_map: dict, judge_model: str, max_tokens: int, checkpoint_dir: Path) -> dict:
    """Dispatch every judge prompt through api_dispatch; return {item_id: int|None}."""
    import asyncio

    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    items = [DispatchItem(item_id=iid, payload=prompt) for iid, (_r, _k, prompt) in id_map.items()]

    def build_request(item: DispatchItem) -> dict:
        return {
            "model": judge_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": item.payload}],
        }

    def parse_response(text: str):
        score = parse_role_score(text)
        if score is None:
            raise ValueError("no 0-3 score parsed")
        return score

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(
        dispatch_calls(
            items,
            model=judge_model,
            build_request=build_request,
            parse_response=parse_response,
            checkpoint_dir=checkpoint_dir,
        )
    )
    return {iid: (res.result if not res.error else None) for iid, res in results.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--responses-dir", required=False)
    ap.add_argument("--roles-dir", required=False)
    ap.add_argument("--output-dir", required=False)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--roles", nargs="+", help="subset of role names")
    ap.add_argument("--stub", action="store_true", help="deterministic score, no API (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="compose only; zero API calls")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        from explore_persona_space.llm.api_dispatch import (  # noqa: F401
            DispatchItem,
            dispatch_calls,
        )
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok")
        return 0

    assert args.responses_dir and args.roles_dir and args.output_dir, (
        "--responses-dir, --roles-dir, --output-dir are required (or --import-check)"
    )
    responses_dir = Path(args.responses_dir)
    roles_dir = Path(args.roles_dir)
    output_dir = Path(args.output_dir)

    id_map, existing = _collect_uncached(responses_dir, roles_dir, output_dir, args.roles)
    _log(f"[judge] {len(id_map)} unscored responses across {len(existing)} roles")

    if args.dry_run:
        _log(f"[judge] dry-run: would judge {len(id_map)} responses with {args.judge_model}")
        return 0
    if not id_map:
        _log("[judge] nothing to score — all cached")
        return 0

    if args.stub:
        raw = {iid: _stub_score(iid) for iid in id_map}
    else:
        raw = _run_api(id_map, args.judge_model, args.max_tokens, output_dir / ".dispatch")

    new_by_role: dict[str, dict[str, int]] = {}
    n_scored = n_dropped = 0
    for iid, (role, key, _prompt) in id_map.items():
        score = raw.get(iid)
        if score is None:  # rule 9: DROP a malformed/failed return, never coerce
            n_dropped += 1
            continue
        new_by_role.setdefault(role, {})[key] = int(score)
        n_scored += 1
    _write_scores(output_dir, existing, new_by_role)
    _log(f"[judge] wrote scores: {n_scored} scored, {n_dropped} dropped (unparsed/failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
