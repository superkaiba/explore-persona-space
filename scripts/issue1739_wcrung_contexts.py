"""Context reconstruction for the #1739 HELD-OUT BATTERY eval rung ("wcrung").

Rebuilds the RENDERED prompts for the #1092 store's EVAL-ONLY rows — the rows
held out of every #1739 fit pool (``constants.STORE_FIT_ROWS`` = 18,793 of
21,193; ``store_io.fit_pool_mask``) — from the three PUBLISHED #1092 corpus
artifacts, under #1092's OWN render convention, so a reconstructed prompt is the
prompt the reused ``prefix_end`` / ``context_end`` summaries were captured
under.

WHAT THE EVAL-ONLY ROWS ACTUALLY ARE (verified against the published artifacts,
not assumed): the #594 50-context battery crossed with the 48 dense-core
queries — 50 x 48 = 2,400, matching BOTH ``manifest_stats.n_eval_only_rows``
and ``rows_by_stratum.battery``, and matching ``store_io.fit_pool_mask``'s
belt-and-suspenders predicate (``is_eval_only`` OR ``stratum == "battery"``).
They are a CONTIGUOUS TAIL of the manifest (rows 18,793..21,192), which is what
makes the summary slice cheap.

The battery is NOT a sample of real chat traffic. Realized family mix:

    persona 14 | wildchat 10 | icl 8 | rephrase 6 | format 5 | behavior 5 | default 2

Only ``wildchat`` (10 real WildChat-1M conversation prefixes: 5 two-turn, 5
eight-turn) and ``default`` (2 bare/helpful-assistant) are GENERIC chat — 12
contexts / 576 rows. The ``behavior`` family DIRECTLY instructs behaviors this
experiment measures (``f8_behav_sycophant`` / ``f8_behav_refusal`` /
``f8_behav_harmful``), and ``icl`` / ``format`` / ``rephrase`` are constructed
elicitation contexts. A "generic chat traffic lacks these behaviors" read MUST
therefore be scoped with ``--families wildchat,default``; the eliciting families
are useful only as a CONTRAST arm (they should show dynamic range, which is what
distinguishes a real generic-chat floor from a rig-wide floor). Selecting the
whole battery and calling it generic chat would invert the finding.

MULTI-TURN PREFIXES ARE THE REASON THIS MODULE EXISTS. The pvsynth staged-context
schema (``generation.context_messages``) wraps a single ``prefix_text`` in ONE
system message, and ``generation.render_prompt_parts`` derives the prefix by
slicing at the FIRST user-turn header. Neither is correct for a battery
``wildchat`` row: its prefix is a real prior CONVERSATION (``prefix_turns``,
2 or 8 alternating turns, no system prompt), #1092 rendered it as actual chat
turns (``issue1092_gpu_phase._render_instruct``), and slicing at the first user
header would cut inside that conversation. So rows here carry ``prefix_turns``
and render through :func:`render_row_prompt`, mirroring #1092 verbatim. This is
capture-convention parity, NOT a new render: the CLAUDE.md persona-injection
rule (persona ALWAYS system role) governs PERSONA prefixes — the battery's
persona/behavior/format/rephrase families do carry system prompts and are
rendered exactly that way, as a single system turn.

PARITY GATE: every reconstructed prompt's token count is checked against the
row's own stored ``n_tokens_instruct``. That field was written by the capture
run, so an exact match across all selected rows is mechanical proof the
reconstruction reproduces the captured render (same tokenizer pin on both sides:
``Qwen/Qwen2.5-7B-Instruct`` @ ``a09a35458c702b33eeacc393d103063234e8bc28``).

CONTENT HYGIENE: logs, digests, and the emitted manifest carry ids, families,
counts, and token lengths — NEVER prompt / prefix / query text. The battery
mixes unscreened real WildChat user text with behavior-instruction contexts, so
reconstructed text stays on disk and is referenced by id + index only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_wcrung_contexts.py"
    assert sentinel.exists(), f"repo-root derivation failed: {sentinel} missing"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE any torch/transformers import (thread caps + credentials)

logger = logging.getLogger("issue1739_wcrung_contexts")

RUNG = "wcrung"
SPLIT = "eval"

# The #594 battery file is IN GIT (the #1092 corpus builder's BATTERY_PATH).
BATTERY_PATH = _REPO_ROOT / "data" / "issue594" / "battery.json"
# #1092 prefixes the battery context id when it mints the manifest prefix_id
# (issue1092_build_corpus: prefix_id = f"batt_{ctx_id}").
BATTERY_PREFIX_TAG = "batt_"

CORPUS_PREFIX = "issue1092_realistic_crossing/corpus"
CORPUS_FILES = ("manifest.jsonl", "prefix_store.jsonl", "query_store.jsonl")

# Genuine generic-chat families — the ONLY families a "generic chat traffic"
# claim may rest on (see module docstring).
GENERIC_CHAT_FAMILIES = ("wildchat", "default")
# Families whose contexts deliberately elicit / structure behavior. Kept
# selectable as an explicit CONTRAST arm, never folded into a generic-chat read.
ELICITING_FAMILIES = ("behavior", "icl", "format", "rephrase", "persona")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--families",
        default=",".join(GENERIC_CHAT_FAMILIES),
        help=(
            "comma-separated battery families to select, or 'all'. Default is the "
            "generic-chat set (wildchat,default). A generic-chat claim MUST NOT "
            "include the eliciting families."
        ),
    )
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_1739/wcrung"))
    ap.add_argument(
        "--stage-root",
        type=Path,
        default=Path("data/issue_1739/wcrung_stage"),
        help="MIRROR ROOT for Hub staging (files land at <root>/<repo-relative path>)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="cap the selected row count (SMOKE, or a deliberate budget cap)",
    )
    ap.add_argument(
        "--no-parity-gate",
        action="store_true",
        help="skip the n_tokens_instruct parity gate (DIAGNOSTIC ONLY)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the REAL branch, then exit 0",
    )
    return ap.parse_args(argv)


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _iter_jsonl(path: Path):
    """Text-mode iteration — never ``splitlines()`` (U+2028 shreds real user text, #825)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def stage_corpus(stage_root: Path) -> dict[str, Path]:
    """Stage the three #1092 corpus artifacts at the PINNED manifest revision.

    All three are pinned at ``CORPUS_MANIFEST_REVISION`` so the manifest and the
    stores it indexes are one coherent snapshot (the row ids must resolve in the
    stores the capture read).
    """
    from explore_persona_space.experiments.issue_1739.constants import CORPUS_MANIFEST_REVISION
    from explore_persona_space.orchestrate import hub

    out: dict[str, Path] = {}
    for name in CORPUS_FILES:
        path_in_repo = f"{CORPUS_PREFIX}/{name}"
        target = stage_root / path_in_repo
        hub.stage_hub_file(
            hub.DEFAULT_DATASET_REPO,
            path_in_repo,
            target,
            revision=CORPUS_MANIFEST_REVISION,
        )
        out[name] = target
    print(
        f"[phase=wcrung_stage] staged {len(out)} corpus artifacts @ {CORPUS_MANIFEST_REVISION[:8]}",
        flush=True,
    )
    return out


def battery_family_by_prefix_id() -> dict[str, str]:
    """``prefix_id -> family`` for the #594 battery, read from the in-git file.

    Mirrors ``issue1092_build_corpus._load_battery`` normalization and its
    ``prefix_id = f"batt_{ctx_id}"`` minting so the keys join the manifest.
    """
    if not BATTERY_PATH.exists():
        raise FileNotFoundError(
            f"#594 battery file not found at {BATTERY_PATH}; it is in git — "
            "check the checkout rather than regenerating it"
        )
    raw = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        contexts = raw.get("instances") or raw.get("contexts") or raw.get("examples")
        if contexts is None:
            raise KeyError(f"battery dict at {BATTERY_PATH} has no instances/contexts/examples")
    else:
        contexts = raw
    out: dict[str, str] = {}
    for i, ctx in enumerate(contexts):
        ctx_id = ctx.get("id") or f"batt_{i:03d}"
        family = ctx.get("family")
        if not family:
            raise ValueError(f"battery context {ctx_id!r} carries no 'family'")
        out[f"{BATTERY_PREFIX_TAG}{ctx_id}"] = str(family)
    return out


def eval_only_rows(manifest_path: Path) -> list[tuple[int, dict]]:
    """``(store_row_index, row)`` for every EVAL-ONLY manifest row.

    ``store_row_index`` is the row's position in manifest order, which IS the
    row axis of the ``{kind}_L{layer}.npy`` summary arrays (#1092's own consumer
    reads row metadata from this manifest; ``store_io`` falls back to it because
    the realized ``cell_*`` dirs carry no row_index sidecars). Verified against
    the published store: each layer file is 21,193 x 3584 fp16.
    """
    from explore_persona_space.experiments.issue_1739.constants import (
        STORE_FIT_ROWS,
        STORE_TOTAL_ROWS,
    )

    rows = list(_iter_jsonl(manifest_path))
    if len(rows) != STORE_TOTAL_ROWS:
        raise ValueError(
            f"manifest row count {len(rows)} != pinned STORE_TOTAL_ROWS {STORE_TOTAL_ROWS}; "
            "the corpus revision and the store pin have diverged"
        )
    # Same predicate as store_io.fit_pool_mask, inverted.
    selected = [
        (i, r)
        for i, r in enumerate(rows)
        if bool(r.get("is_eval_only")) or r.get("stratum") == "battery"
    ]
    n_eval = len(selected)
    if n_eval != STORE_TOTAL_ROWS - STORE_FIT_ROWS:
        raise ValueError(
            f"eval-only row count {n_eval} != STORE_TOTAL_ROWS - STORE_FIT_ROWS "
            f"({STORE_TOTAL_ROWS - STORE_FIT_ROWS}); the fit/eval split has moved"
        )
    return selected


def _prefix_turns(prefix_item: dict) -> list[dict]:
    """Prefix turns for a battery prefix entry, mirroring #1092's reader.

    A persona/behavior/format/rephrase battery context carries a single system
    turn; a wildchat context carries the real 2- or 8-turn conversation; the
    bare ``default`` context carries none.
    """
    turns = prefix_item.get("prefix_turns")
    if turns is None:
        raise ValueError(f"prefix {prefix_item.get('prefix_id')!r} has no 'prefix_turns'")
    out = []
    for t in turns:
        role, content = t.get("role"), t.get("content")
        if role is None or content is None:
            raise ValueError(f"prefix {prefix_item.get('prefix_id')!r} has a malformed turn")
        out.append({"role": str(role), "content": str(content)})
    return out


def _query_text(query_item: dict) -> str:
    for key in ("query", "text", "content", "query_text"):
        val = query_item.get(key)
        if isinstance(val, str) and val:
            return val
    raise ValueError(f"query {query_item.get('query_id')!r} carries no text field")


def render_row_prompt(tokenizer, turns: list[dict], query: str) -> tuple[str, str]:
    """``(prefix_text, prompt_text)`` under #1092's INSTRUCT render convention.

    Mirrors ``issue1092_gpu_phase._render_instruct`` (prompt) and
    ``_render_prefix_instruct`` (prefix) verbatim — including the empty-turns
    guard, where the Qwen template cannot render an empty message list and the
    prefix is instead the template-injected system block sliced off the rendered
    prompt at its user-turn header (#1092 round-8.2).

    Deliberately NOT ``generation.render_prompt_parts``: that helper always
    slices at the FIRST user-turn header, which cuts inside a multi-turn
    conversation prefix.
    """
    from explore_persona_space.experiments.issue_1739.generation import INSTRUCT_USER_HEADER

    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    messages.append({"role": "user", "content": query})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if turns:
        prefix = tokenizer.apply_chat_template(
            [{"role": t["role"], "content": t["content"]} for t in turns],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        idx = prompt.find(INSTRUCT_USER_HEADER)
        if idx < 0:
            raise ValueError(
                "bare-context instruct render lacks a user-turn header; cannot "
                "derive the prefix (template drift?)"
            )
        prefix = prompt[:idx]
    return prefix, prompt


def reconstruct_contexts(
    *,
    corpus: dict[str, Path],
    families: set[str] | None,
    max_rows: int | None,
    tokenizer,
    parity_gate: bool = True,
) -> tuple[list[dict], dict]:
    """Rebuild staged-context rows for the selected eval-only battery rows.

    Returns ``(rows, digest)``. Each row carries ``prefix_turns`` (the render
    input), ``query``, the rendered ``prompt``, ``store_row_index`` (the free
    summary slice key), plus ``family`` / ``group_key`` for scoping + folds.
    The digest carries counts, family mix, and token stats — never text.
    """
    fam_by_prefix = battery_family_by_prefix_id()
    prefix_store = {r["prefix_id"]: r for r in _iter_jsonl(corpus["prefix_store.jsonl"])}
    query_store = {r["query_id"]: r for r in _iter_jsonl(corpus["query_store.jsonl"])}
    selected = eval_only_rows(corpus["manifest.jsonl"])

    unknown = {r["prefix_id"] for _, r in selected} - set(fam_by_prefix)
    if unknown:
        raise ValueError(
            f"{len(unknown)} eval-only prefix_id(s) absent from the #594 battery "
            f"(e.g. {sorted(unknown)[:3]}); the battery file and the corpus have diverged"
        )

    rows: list[dict] = []
    parity_fail: list[dict] = []
    token_lens: list[int] = []
    for store_idx, row in selected:
        family = fam_by_prefix[row["prefix_id"]]
        if families is not None and family not in families:
            continue
        turns = _prefix_turns(prefix_store[row["prefix_id"]])
        query = _query_text(query_store[row["query_id"]])
        prefix_text, prompt = render_row_prompt(tokenizer, turns, query)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        token_lens.append(n_tok)
        expected = row.get("n_tokens_instruct")
        if parity_gate and expected is not None and int(expected) != n_tok:
            parity_fail.append(
                {
                    "row_id": row["row_id"],
                    "store_row_index": store_idx,
                    "expected_n_tokens_instruct": int(expected),
                    "rendered_n_tokens": n_tok,
                }
            )
        rows.append(
            {
                "context_id": f"{RUNG}-{row['row_id']}",
                "row_id": row["row_id"],
                "store_row_index": store_idx,
                "family": family,
                "prefix_id": row["prefix_id"],
                "query_id": row["query_id"],
                "prefix_turns": turns,
                "prefix_text": prefix_text,
                "query": query,
                "prompt": prompt,
                "n_tokens_instruct": n_tok,
                "split": SPLIT,
                "rung": RUNG,
                # Group folds by battery CONTEXT (the prefix) — the 48 queries
                # under one prefix are not independent units.
                "group_key": row["prefix_id"],
            }
        )

    if parity_gate and parity_fail:
        raise ValueError(
            f"render parity gate FAILED for {len(parity_fail)} of {len(rows)} rows: the "
            f"reconstructed prompt's token count differs from the row's stored "
            f"n_tokens_instruct, so the reconstruction does NOT reproduce the captured "
            f"render. First mismatches: {parity_fail[:3]}"
        )

    if not rows:
        raise ValueError(
            f"zero rows selected for families={sorted(families) if families else 'all'}; "
            "check --families against the realized battery family mix"
        )

    # Cap AFTER the parity gate so the gate always covers the full selection.
    n_before_cap = len(rows)
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]

    ids = [r["context_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate wcrung context_id")
    idxs = [r["store_row_index"] for r in rows]
    if len(set(idxs)) != len(idxs):
        raise ValueError("duplicate store_row_index — summary slicing would double-count")

    fam_mix = Counter(r["family"] for r in rows)
    generic_only = set(fam_mix) <= set(GENERIC_CHAT_FAMILIES)
    digest = {
        "rung": RUNG,
        "split": SPLIT,
        "n_rows": len(rows),
        "n_rows_before_cap": n_before_cap,
        "n_eval_only_available": len(selected),
        "family_mix": dict(sorted(fam_mix.items())),
        "n_contexts": len({r["prefix_id"] for r in rows}),
        "n_queries": len({r["query_id"] for r in rows}),
        "store_row_index_min": min(idxs),
        "store_row_index_max": max(idxs),
        "token_len_min": min(token_lens),
        "token_len_max": max(token_lens),
        "parity_gate": "skipped" if not parity_gate else "PASS",
        "n_parity_mismatch": len(parity_fail),
        # The load-bearing scoping flag: a generic-chat claim requires True.
        "generic_chat_only": generic_only,
        "eliciting_families_present": sorted(set(fam_mix) & set(ELICITING_FAMILIES)),
    }
    return rows, digest


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.import_check:
        from explore_persona_space.experiments.issue_1739 import (  # noqa: F401
            constants,
            generation,
            store_io,
        )
        from explore_persona_space.experiments.issue_1739.constants import (  # noqa: F401
            CORPUS_MANIFEST_REVISION,
            STORE_FIT_ROWS,
            STORE_TOTAL_ROWS,
        )
        from explore_persona_space.experiments.issue_1739.generation import (  # noqa: F401
            INSTRUCT_USER_HEADER,
            get_tokenizer,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            DEFAULT_DATASET_REPO,
            stage_hub_file,
        )

        print("[import-check] OK: all deferred imports resolved", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return 0

    from explore_persona_space.experiments.issue_1739.generation import get_tokenizer

    families: set[str] | None
    if args.families.strip().lower() == "all":
        families = None
    else:
        families = {f.strip() for f in args.families.split(",") if f.strip()}

    corpus = stage_corpus(args.stage_root)
    tokenizer = get_tokenizer()
    rows, digest = reconstruct_contexts(
        corpus=corpus,
        families=families,
        max_rows=args.max_rows,
        tokenizer=tokenizer,
        parity_gate=not args.no_parity_gate,
    )
    digest["git_commit"] = _git_commit()
    digest["families_requested"] = "all" if families is None else sorted(families)

    _write_json_atomic(args.out_root / "contexts" / f"{RUNG}.json", {"rows": rows, **digest})
    _write_json_atomic(args.out_root / "contexts" / f"{RUNG}_digest.json", digest)

    print(
        f"[phase=wcrung_contexts] rows={digest['n_rows']} contexts={digest['n_contexts']} "
        f"queries={digest['n_queries']} families={digest['family_mix']} "
        f"generic_chat_only={digest['generic_chat_only']} "
        f"parity={digest['parity_gate']} "
        f"tok=[{digest['token_len_min']},{digest['token_len_max']}]",
        flush=True,
    )
    if not digest["generic_chat_only"]:
        print(
            "[phase=wcrung_contexts] NOTE eliciting families present "
            f"({digest['eliciting_families_present']}) — this selection MUST NOT be "
            "reported as a generic-chat read; scope with --families wildchat,default",
            flush=True,
        )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
