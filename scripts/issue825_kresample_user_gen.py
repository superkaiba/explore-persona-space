#!/usr/bin/env python
"""Issue #825 follow-up ``kresample-user`` — Phase 1: fresh Haiku user-turn redraws.

Dan's question on the #825 user-turn null: "how diverse are user answers? r2 among
resampled user answers I guess sets the floor for this." This phase resamples the
Haiku-as-user generator K=4 times per context over the onpolicy instruct-allowlist
contexts, so Phase 3 can compute the inter-resample agreement that CEILINGS any
context->user-answer map (the parent Haiku-u2 user map reads R^2(L19) -1.4272
instruct / -1.4894 pretrained; the onpolicy Qwen-on-policy anchor -0.7689 / -1.8399).

THE ONE VARIABLE vs the anchor targets: nothing about the map is refit here — this
phase only measures the TARGET's within-context resample variance. The generator,
prompt builder, persona briefs, temperature, and model pin are the PARENT Haiku
recipe (``issue825_gen_conversations._haiku_user_turn``) reused verbatim; only the
API call site changes (routed through ``llm.api_dispatch`` per the throughput rule,
never the parent's hand-rolled ThreadPoolExecutor).

Workload: len(instruct allowlist)=1914 contexts x K=4 draws = 7656 Haiku calls,
max_tokens=512 out. Sync fan-out via api_dispatch (Haiku ~45-90s at N~=10k per
docs/api_throughput_guidelines.md; far below the ~500k batch crossover).

TEXT DISCIPLINE: u1/a1 are real-corpus-derived and u2 draws are model-generated
user text. This script NEVER prints turn text — only counts, keep-rates, hashes.
Raw draws are persisted to JSONL (the artifact) and uploaded to the round's OWN
HF prefix ``issue825_kresample_user/`` before Phase 2.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.experiments.issue_825.common import HF_DATA_REPO
from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls
from explore_persona_space.orchestrate.env import load_dotenv

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Parent Haiku recipe constants (model pin + persona briefs) imported from the
# source so they cannot drift. USER_BRIEFS is [(brief_id, brief_text), ...].
from issue825_gen_conversations import HAIKU_GEN_MODEL, USER_BRIEFS  # noqa: E402

HF_PREFIX = "issue825_kresample_user"
K_DRAWS = 4
U2_MAX_TOKENS = 512
U2_TEMPERATURE = 1.0
ALLOWLIST_KEY = "M_instruct_user_chat"  # the brief's n~=1914 context set
_BRIEF_TEXT = {bid: text for bid, text in USER_BRIEFS}


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _build_prompt(u1: str, a1: str, brief_text: str) -> tuple[str, str]:
    """VERBATIM reproduction of ``issue825_gen_conversations._haiku_user_turn``'s
    system+content construction (that module's lines 291-299). Returns
    ``(system, content)``; the caller lifts ``system`` to the top-level Messages
    API param (there is no system message role)."""
    system = (
        "You are simulating a human USER in an ongoing conversation with an AI assistant, "
        f"generating research data. {brief_text} Write ONLY the user's next message: "
        "natural, first-person, no role labels, no meta-commentary."
    )
    content = (
        f"Conversation so far:\n\n[USER]\n{u1}\n\n[ASSISTANT]\n{a1}\n\n"
        "Now write the user's next message."
    )
    return system, content


def _build_request(item: DispatchItem) -> dict:
    p = item.payload
    system, content = _build_prompt(p["u1"], p["a1"], p["brief_text"])
    return {
        "model": HAIKU_GEN_MODEL,  # generation of a simulated user turn, NOT a judge
        "max_tokens": U2_MAX_TOKENS,
        "temperature": U2_TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }


def _parse(text: str) -> str:
    return text.strip()


def _load_convs(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert r.get("u1") and r.get("a1"), f"row {r.get('conv_id')} missing u1/a1"
            out[int(r["conv_id"])] = r
    assert out, f"no conversations in {path}"
    return out


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--conversations",
        type=Path,
        default=Path("data/issue_825/kresample_user/inputs/conversations.jsonl"),
    )
    ap.add_argument(
        "--allowlists",
        type=Path,
        default=Path("data/issue_825/kresample_user/inputs/row_allowlists.json"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/issue_825/kresample_user"))
    ap.add_argument("--k", type=int, default=K_DRAWS)
    ap.add_argument("--smoke", action="store_true", help="tiny slice; no HF upload")
    ap.add_argument("--smoke-n", type=int, default=3)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    convs = _load_convs(args.conversations)
    allow = json.loads(args.allowlists.read_text())[ALLOWLIST_KEY]
    conv_ids = [int(c) for c in allow]
    assert all(c in convs for c in conv_ids), "allowlist conv_id missing from corpus"
    if args.smoke:
        conv_ids = conv_ids[: args.smoke_n]
    print(f"[gen] contexts={len(conv_ids)} K={args.k} -> {len(conv_ids) * args.k} Haiku calls")

    items: list[DispatchItem] = []
    for cid in conv_ids:
        r = convs[cid]
        brief_id = r["brief_id"]
        brief_text = _BRIEF_TEXT[brief_id]
        payload = {"u1": r["u1"], "a1": r["a1"], "brief_text": brief_text, "brief_id": brief_id}
        for k in range(1, args.k + 1):
            items.append(DispatchItem(item_id=f"c{cid}_d{k}", payload=payload))

    cache_dir = args.out_dir / "gen_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = asyncio.run(
        dispatch_calls(
            items,
            model=HAIKU_GEN_MODEL,
            build_request=_build_request,
            parse_response=_parse,
            response_valid=lambda t: isinstance(t, str) and len(t.strip()) > 0,
            cache_dir=cache_dir,
            force_path="sync",
        )
    )
    wall = time.time() - t0

    rows: list[dict] = []
    n_ok = n_err = 0
    for cid in conv_ids:
        for k in range(1, args.k + 1):
            res = results[f"c{cid}_d{k}"]
            if res.error or not isinstance(res.result, str) or not res.result.strip():
                n_err += 1
                rows.append(
                    {
                        "conv_id": cid,
                        "draw": k,
                        "u2": None,
                        "error": True,
                        "reason": res.reason,
                        "category": res.category,
                    }
                )
            else:
                n_ok += 1
                rows.append(
                    {
                        "conv_id": cid,
                        "draw": k,
                        "u2": res.result,
                        "brief_id": convs[cid]["brief_id"],
                        "error": False,
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / ("draws_smoke.jsonl" if args.smoke else "draws.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # digest-only integrity read (no turn text)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()

    # contexts where ALL K draws succeeded (usable for the floor)
    per_ctx_ok: dict[int, int] = {}
    for row in rows:
        if not row["error"]:
            per_ctx_ok[row["conv_id"]] = per_ctx_ok.get(row["conv_id"], 0) + 1
    n_full = sum(1 for c in conv_ids if per_ctx_ok.get(c, 0) == args.k)

    meta = {
        "followup_label": "kresample-user",
        "phase": "gen",
        "allowlist_key": ALLOWLIST_KEY,
        "n_contexts": len(conv_ids),
        "k_draws": args.k,
        "n_calls": len(items),
        "n_ok": n_ok,
        "n_err": n_err,
        "n_contexts_all_k_ok": n_full,
        "generator_model": HAIKU_GEN_MODEL,
        "temperature": U2_TEMPERATURE,
        "max_tokens": U2_MAX_TOKENS,
        "prompt_builder": "issue825_gen_conversations._haiku_user_turn (verbatim)",
        "persona_brief": "per-context row brief_id (parent-assigned), verbatim USER_BRIEFS text",
        "routing": "llm.api_dispatch.dispatch_calls force_path=sync",
        "draws_jsonl_sha256": sha,
        "wall_seconds": round(wall, 1),
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": args.smoke,
    }
    meta_path = args.out_dir / ("draws_smoke_meta.json" if args.smoke else "draws_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[gen] done: ok={n_ok} err={n_err} contexts_all_k_ok={n_full}/{len(conv_ids)} "
        f"wall={wall:.1f}s sha={sha[:12]} -> {out_path}"
    )

    if args.smoke or args.no_upload:
        print("[gen] skipping HF upload")
        return
    from explore_persona_space.orchestrate import hub

    hub._upload(
        out_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/raw_completions/draws.jsonl",
        upload_as_file=True,
    )
    hub._upload(
        meta_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{HF_PREFIX}/raw_completions/draws_meta.json",
        upload_as_file=True,
    )
    print(f"[gen] uploaded draws + meta -> {HF_DATA_REPO}/{HF_PREFIX}/raw_completions/")


if __name__ == "__main__":
    main()
