#!/usr/bin/env python
"""Issue #1345 on-policy-vs-injected program — ON-POLICY ANSWER generation.

The injected arms embed a FIXED verbatim answer (the parent track-S response)
into a wrapper the model writes; these arms invert that: the wrapper is fixed
and the MODEL writes the answer. Same conv_id space, so every on-policy row is
data-paired with its injected twin and with the comparator stores.

THREE shapes, ONE prompt-construction rule
------------------------------------------
The generation prompt is ALWAYS the CONSUMER's own render truncated at the
answer's start — never a re-rendered approximation, never a chat template whose
preamble the capture does not carry. So the only variable between an injected
arm and its on-policy twin is WHO WROTE THE ANSWER:

  bare_text   `User: {q}\\n\\nAssistant: `  — segments[:4] of the naturalistic
              render (`c._single_turn_segments(..., chat=False)`). Consumed as a
              comparator-convs JSONL by `--comparator no_template`.
  chat        `<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n` —
              segments[:4] of the chat render. NOT `apply_chat_template`, which
              would prepend Qwen's default-system preamble the capture render
              does not carry (the #1776 preamble class). Consumed by
              `--comparator chat`.
  story_slot  the V1 story text BYTE-SLICED at the stored `a_start` — the
              instruct-written narrative + attribution up to (and including) the
              opening quote, with the answer slot left for the measured model.
              Consumed as a kept-stories JSONL by the capture's ARM path.

Answer conventions
------------------
* bare_text / chat stop at the next turn marker; story_slot stops at the closing
  double quote — which IS the V1 answer convention (`match_verbatim_turn` finds
  the answer between the attribution's quotes, so a V1-embeddable answer never
  contains one). A run to the token cap leaves the quote unclosed; the capture's
  V1 closer reproduces it either way.
* One trailing `"` is stripped (recorded per row) so the arm's closer adds it
  back exactly once — the V1 span convention (`a_end` sits BEFORE the quote).
* `ANSWER_CHAR_MIN` drops degenerate near-empty answers (the #825 zero-width
  span class) instead of letting the extractor's `1 <= s < e` assert kill a
  mid-GPU run.

Gen-time span validation (the #825 rule)
----------------------------------------
Every kept row is rendered through the CONSUMER's OWN render function
(`render_comparator_turn` / `render_boundary_turn`) at generation time, with the
real tokenizer — not a mirrored re-implementation that can drift. A row whose
render returns None is DROPPED and counted, so a span bug surfaces here at
CPU cost instead of on a billed pod.

Variant scoping
---------------
Each (shape x model) cell runs under its OWN `EPM_I1345_VARIANT` (registered in
`ONPOLICY_ANSWER_VARIANTS` below), which scopes every output dir + HF prefix via
`issue1345_common._VSUB`. The env contract is asserted at entry: an unregistered
variant, or one whose (shape, model) disagrees with the CLI, refuses to run.

CLI:
  uv run python scripts/issue1345_onpolicy_answers_gen.py --shape bare_text --model instruct
  uv run python scripts/issue1345_onpolicy_answers_gen.py --shape story_slot --model pretrained
  uv run python scripts/issue1345_onpolicy_answers_gen.py --shape chat --model pretrained \\
      --verify-pool          # zero-GPU CPU preflight (pool + prompts + fingerprint)
  uv run python scripts/issue1345_onpolicy_answers_gen.py --import-check
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# vLLM reads this at IMPORT time. `main()` loads a tokenizer (CUDA-adjacent)
# before building the engine, so the default `fork()` duplicates poisoned parent
# state into EngineCore, which dies 1-4 s after init with no traceback of its own
# (#628). Set HERE rather than relying on the transitive `gp` import: a future
# import cleanup would silently reopen the class.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_boundary_ablation_gen as bg  # noqa: E402 — round constants only
import issue1345_common as c  # noqa: E402
import issue1345_gen_stories as g  # noqa: E402 — HF/upload + yield-floor helpers
import issue1345_gen_stories_paired as gp  # noqa: E402 — matched-n seed pool

# Shared-node vLLM util recipe (#1902 crash 1), hoisted to the shared module by
# #1942 (this file's local copy was a deliberate re-port while #1902's branch
# was unmerged). This round runs on exclusive hosts → EXCLUSIVE_HOST_UTIL_CAP.
from explore_persona_space.eval.vllm_util import (  # noqa: E402
    EXCLUSIVE_HOST_UTIL_CAP,
    resolve_vllm_util,
)

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
SHAPE_BARE = "bare_text"
SHAPE_CHAT = "chat"
SHAPE_STORY_SLOT = "story_slot"
SHAPES = (SHAPE_BARE, SHAPE_CHAT, SHAPE_STORY_SLOT)

# File-stem slug per shape (distinct per shape; the variant scopes the dir).
SHAPE_SLUG = {SHAPE_BARE: "op_ntpl", SHAPE_CHAT: "op_chat", SHAPE_STORY_SLOT: "op_slot"}

# Which capture render consumes each shape's rows (documented in the manifest).
SHAPE_CONSUMER = {
    SHAPE_BARE: "capture --comparator no_template --convs-jsonl <rows>",
    SHAPE_CHAT: "capture --comparator chat --convs-jsonl <rows>",
    SHAPE_STORY_SLOT: "capture --arm <story-slot arm> (kept-stories shape)",
}

# The comparator key whose render validates a shape's rows at gen time.
SHAPE_COMPARATOR = {SHAPE_BARE: "no_template", SHAPE_CHAT: "chat"}

# Stop strings. bare_text/chat stop at the next turn marker (the BASE model has
# no template eos and would otherwise keep writing both sides of the dialogue);
# story_slot stops at the closing quote — the V1 answer convention.
SHAPE_STOP = {
    SHAPE_BARE: ["\n\nUser:", "\nUser:"],
    SHAPE_CHAT: ["<|im_end|>", "<|im_start|>"],
    SHAPE_STORY_SLOT: ['"'],
}

# Free-generation budget (the project default for a free answer) + an engine
# window sized to it (gotchas.md: max_model_len tracks max_new_tokens).
ONPOLICY_MAX_NEW_TOKENS = c.STORY_MAX_NEW_TOKENS
MAX_MODEL_LEN = g.MAX_MODEL_LEN
PROMPT_TOKEN_BUDGET = MAX_MODEL_LEN - ONPOLICY_MAX_NEW_TOKENS - 64
ANSWER_CHAR_MIN = gp.ANSWER_CHAR_MIN

# Smoke slice: enough rows that the validation + drop-accounting paths run, and
# the yield floor resolves to 1 so ANY kept row proceeds (g.resolve_yield_floor).
SMOKE_N_ROWS = 4

# Placeholder answer used ONLY to make the 6-segment render list well-formed
# (`rf._present_turns` requires a truthy a1); segments[:4] never contain it.
_SEGMENT_PROBE_ANSWER = "PROBE"


# ---------------------------------------------------------------------------
# Variant registry + env contract
# ---------------------------------------------------------------------------
# One (shape x measured-model) cell per variant, so each cell's output dir and
# HF prefix are disjoint (`c._VSUB`). Registered HERE rather than in the shared
# variant tuples: these arms drive no REGIMES / STORY_REGIME / R4_MODELS gate,
# so the shared module stays byte-untouched and the three live rounds'
# fingerprints cannot move.
ONPOLICY_ANSWER_VARIANTS: dict[str, tuple[str, str]] = {
    "onpolicy_answers_ntpl_instruct": (SHAPE_BARE, "instruct"),
    "onpolicy_answers_ntpl_base": (SHAPE_BARE, "pretrained"),
    "onpolicy_answers_chat_instruct": (SHAPE_CHAT, "instruct"),
    "onpolicy_answers_chat_base": (SHAPE_CHAT, "pretrained"),
    "onpolicy_answers_slot_instruct": (SHAPE_STORY_SLOT, "instruct"),
    "onpolicy_answers_slot_base": (SHAPE_STORY_SLOT, "pretrained"),
}

# The character whose attribution the V1 story prefixes carry (story_slot's
# default source bundle). Read from the boundary-ablation round's own constant
# rather than re-typed: the V1 anchor is that round's `Assistant`-named corpus,
# and a re-typed literal is the "never type a value from memory" trap.
V1_SOURCE_CHARACTER = bg.ROUND_CHARACTER


def assert_onpolicy_env(shape: str, model: str) -> None:
    """Refuse to run outside a REGISTERED (shape x model) variant scope.

    The variant scopes every output dir + HF prefix; running a cell under the
    wrong one silently writes another cell's paths. Fail at entry, never after
    a pod has generated.
    """
    assert c.VARIANT in ONPOLICY_ANSWER_VARIANTS, (
        f"EPM_I1345_VARIANT={c.VARIANT!r} is not a registered on-policy answer variant — "
        f"expected one of {sorted(ONPOLICY_ANSWER_VARIANTS)}"
    )
    want_shape, want_model = ONPOLICY_ANSWER_VARIANTS[c.VARIANT]
    assert (shape, model) == (want_shape, want_model), (
        f"variant {c.VARIANT!r} is registered for shape={want_shape!r} model={want_model!r} "
        f"but this run passed shape={shape!r} model={model!r} — the variant scopes the "
        "output dir + HF prefix, so a mismatch writes another cell's paths"
    )


# ---------------------------------------------------------------------------
# Prompt construction — the consumer's own render, truncated at the answer
# ---------------------------------------------------------------------------
def answer_slot_prefix(question: str, *, chat: bool) -> str:
    """The capture render's own text up to (not including) the answer.

    Built through `c._single_turn_segments` — the SAME idiom
    `render_comparator_turn` uses — so the generation context is byte-identical
    to the captured context's prefix by construction.
    """
    segs = c._single_turn_segments({"u1": question, "a1": _SEGMENT_PROBE_ANSWER}, chat=chat)
    assert len(segs) == 6, (
        f"expected 6 single-turn segments, got {len(segs)} — track-S contract drift"
    )
    prefix = "".join(segs[:4])
    assert _SEGMENT_PROBE_ANSWER not in prefix, (
        "the probe answer leaked into the generation prefix — segment order drift"
    )
    return prefix


def build_gen_prompt(row: dict, *, shape: str) -> str:
    """Generation prompt for one pool row (per-shape, one rule)."""
    if shape == SHAPE_STORY_SLOT:
        return row["prefix"]
    return answer_slot_prefix(row["question"], chat=shape == SHAPE_CHAT)


# ---------------------------------------------------------------------------
# Seed pools
# ---------------------------------------------------------------------------
def load_comparator_pool(matched_dir: Path, dl_dir: Path) -> tuple[list[dict], dict]:
    """Matched-n allowlist x parent track-S questions (the shared conv space).

    The SAME pool loader every injected arm uses, so the on-policy comparator
    rows are conv_id-paired with every other arm by construction.
    """
    pool, counts = gp.load_paired_pool(matched_dir, dl_dir)
    return [{"conv_id": r["conv_id"], "question": r["question"]} for r in pool], dict(counts)


def _turn_boundary_end(turn: dict) -> int:
    """`boundary_end` under either name (the paired round calls it marker_end)."""
    if "boundary_end" in turn:
        return int(turn["boundary_end"])
    assert "marker_end" in turn, f"turn carries neither boundary_end nor marker_end: {sorted(turn)}"
    return int(turn["marker_end"])


def load_story_slot_pool(dl_dir: Path, stories_jsonl: Path | None) -> tuple[list[dict], dict]:
    """Story prefixes byte-sliced from a kept-stories bundle at the stored a_start.

    Default source is the V1 anchor's sha-pinned parent bundle (the capture's own
    `load_v1_stories`, which verifies the sha + row count and normalizes the turn
    key names). `--stories-jsonl` points at any kept-stories JSONL so a character
    arm's own stories can drive the same shape.
    """
    if stories_jsonl is None:
        import issue1345_boundary_ablation_capture as cap

        rows = cap.load_v1_stories(dl_dir)
        source = f"V1 pinned bundle ({cap.V1_KEPT_HF_PATH})"
        assert c.STORY_CHARACTER_NAME == V1_SOURCE_CHARACTER, (
            f"EPM_STORY_CHARACTER_NAME={c.STORY_CHARACTER_NAME!r} but the V1 story prefixes "
            f"carry {V1_SOURCE_CHARACTER!r}'s attribution — the capture's gate is built from "
            "the env name, so a mismatch fails at the extraction trust boundary"
        )
    else:
        rows = c.read_jsonl(stories_jsonl)
        source = str(stories_jsonl)
    assert rows, f"no kept stories in {source}"

    counts = {"stories": len(rows), "kept": 0, "multi_turn": 0, "span_not_verbatim": 0}
    pool: list[dict] = []
    for r in rows:
        turns = r["parsed_turns"]
        if len(turns) != 1:
            counts["multi_turn"] += 1
            continue
        turn = dict(turns[0])
        turn["boundary_end"] = _turn_boundary_end(turn)
        a_start, a_end = int(turn["a_start"]), int(turn["a_end"])
        # Trust boundary (fail-loud, matching the capture's own re-verify): the
        # stored span MUST be the verbatim answer, else a_start is not the
        # answer slot and every prefix would be mis-sliced.
        if c.norm_text(r["story"][a_start:a_end]) != c.norm_text(r["answer"]):
            counts["span_not_verbatim"] += 1
            continue
        pool.append(
            {
                "conv_id": str(r["conv_id"]),
                "prefix": r["story"][:a_start],
                "turn": turn,
                "source_story": r["story"],
                "source_answer": r["answer"],
            }
        )
        counts["kept"] += 1
    assert not counts["span_not_verbatim"], (
        f"{counts['span_not_verbatim']} source stories have a non-verbatim stored span — "
        f"bundle drift in {source}, refusing to slice prefixes off it"
    )
    assert pool, f"story-slot pool empty after filtering {source}"
    print(f"[seeds] story-slot pool from {source}: {counts}", flush=True)
    return sorted(pool, key=lambda r: r["conv_id"]), counts


def filter_pool_by_prompt_budget(
    pool: list[dict], tokenizer, *, shape: str
) -> tuple[list[dict], dict]:
    """Drop rows whose generation prompt exceeds the engine's prompt budget.

    Load-time length validation (#952): one over-budget prompt is a hard vLLM
    `add_request` ValueError that kills the whole engine mid-production.
    """
    kept, counts = [], {"prompt_over_budget": 0, "max_prompt_tokens": 0}
    for row in pool:
        prompt = build_gen_prompt(row, shape=shape)
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_tok > PROMPT_TOKEN_BUDGET:
            counts["prompt_over_budget"] += 1
            continue
        # MEASURED max, recorded in the yield report + manifest: the budget is
        # satisfied-by-headroom rather than by a frozen literal, so a corpus
        # change re-measures itself instead of silently eating the margin.
        counts["max_prompt_tokens"] = max(counts["max_prompt_tokens"], n_tok)
        kept.append(row)
    print(
        f"[seeds] prompt-budget filter ({shape}, budget {PROMPT_TOKEN_BUDGET} tok): "
        f"kept {len(kept)}/{len(pool)} {counts}",
        flush=True,
    )
    assert kept, f"every {shape} prompt exceeded the {PROMPT_TOKEN_BUDGET}-token budget"
    return kept, counts


# ---------------------------------------------------------------------------
# Fingerprint (content key over everything that determines the rows)
# ---------------------------------------------------------------------------
def bundle_fingerprint(shape: str, model_key: str, row_ids: list[str]) -> str:
    """Content key over shape + model + prompt recipe + budgets + row ids.

    Hashes the SOURCE of the prompt builders (`answer_slot_prefix` +
    `c._single_turn_segments`), so a recipe edit invalidates a resume instead of
    silently mixing two prompt constructions in one bundle.
    """
    payload = json.dumps(
        {
            "shape": shape,
            "model": model_key,
            "variant": c.VARIANT,
            "character": c.STORY_CHARACTER_NAME,
            "max_new_tokens": ONPOLICY_MAX_NEW_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "prompt_token_budget": PROMPT_TOKEN_BUDGET,
            "answer_char_min": ANSWER_CHAR_MIN,
            "temperature": c.STORY_TEMPERATURE,
            "stop": SHAPE_STOP[shape],
            "prompt_recipe_sha": hashlib.sha256(
                (
                    inspect.getsource(answer_slot_prefix)
                    + inspect.getsource(build_gen_prompt)
                    + inspect.getsource(c._single_turn_segments)
                ).encode()
            ).hexdigest(),
            "rows_sha": hashlib.sha256("\n".join(sorted(row_ids)).encode()).hexdigest(),
            "n_rows": len(row_ids),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Row assembly + gen-time validation through the CONSUMER's own render
# ---------------------------------------------------------------------------
def _strip_one_trailing_quote(text: str) -> tuple[str, bool]:
    """Drop at most ONE trailing double quote (the V1 `a_end`-before-quote rule)."""
    out = text.rstrip()
    if out.endswith('"'):
        return out[:-1].rstrip(), True
    return out, False


_V1_GATE = None


def _v1_gate_cached():
    """The capture's own V1 gate, imported once (not per row).

    Deliberately the CONSUMER's function rather than a re-derived check: a
    re-derived copy is free to drift from the gate that will actually assert.
    """
    global _V1_GATE
    if _V1_GATE is None:
        import issue1345_boundary_ablation_capture as cap

        _V1_GATE = cap.gate_for_capture(cap.V1_ARM)
    return _V1_GATE


def assemble_row(
    pool_row: dict, answer_text: str, *, shape: str, model_key: str
) -> tuple[dict | None, str]:
    """One emitted row in the CONSUMER's own schema, plus a drop reason.

    Returns ``(row, "ok")`` or ``(None, <named reason>)`` so `keep_rows` can
    account every drop class separately instead of collapsing them.
    """
    if shape == SHAPE_STORY_SLOT:
        answer, stripped = _strip_one_trailing_quote(answer_text)
        # LEADING whitespace must come off before a_start is taken. The parent V1
        # convention puts a_start at the answer's first CONTENT character — the
        # gate re-derives it by normalized occurrence search, and its own comment
        # says a space between the opening quote and the match start "belongs to
        # neither". Storing a_start = len(prefix) with a space-initial answer
        # points the span AT the space, so the capture's span-consistency assert
        # sees a +1 disagreement and dies. Measured 7/2089 (0.34%) on the real
        # pool, and ZERO of the 2,082 space-free rows disagree — the split is
        # exact, so this is the whole class. The store's Y spans are what the
        # fits read, so the on-policy and injected arms MUST encode the answer
        # region under one convention, not merely self-consistently.
        answer, lead_ws = answer.lstrip(), len(answer) - len(answer.lstrip())
        if len(answer) < ANSWER_CHAR_MIN:
            return None, "answer_too_short"
        prefix = pool_row["prefix"]
        turn = dict(pool_row["turn"])
        a_start = int(turn["a_start"])
        turn["a_end"] = a_start + len(answer)
        # The story text CLOSES the quote (a_end stays BEFORE it, the V1 span
        # convention). Load-bearing for the consumer, free at render time:
        # `render_boundary_turn` truncates at a_end and re-adds the arm's own
        # closer, so the rendered text is identical either way — but the ARM
        # path first re-gates the row with the V1 gate, which REFUSES an
        # unclosed answer quote (`answer_quote_not_closed`).
        story = prefix + answer + '"'
        # Byte-derivation invariant: the on-policy story's prefix IS the source
        # story's prefix, so q/boundary/a_start carry over unchanged.
        assert story[:a_start] == pool_row["source_story"][:a_start] == prefix, (
            f"{pool_row['conv_id']}: on-policy story prefix drifted from the source slice"
        )
        assert story[a_start : turn["a_end"]] == answer, (
            f"{pool_row['conv_id']}: answer span does not reproduce the generated answer"
        )
        assert story[turn["a_end"] :] == '"', (
            f"{pool_row['conv_id']}: the story must close the answer quote exactly once"
        )
        # The ARM path re-gates every row with the V1 answer-ANCHORED gate, which
        # refuses `answer_occurrences_multi` — and `render_arm`'s re-gate is an
        # ASSERT, not a skip, so ONE multi-occurrence row kills the whole capture
        # mid-GPU-run. The answer always sits at the story tail here, so the only
        # realistic collision is a short answer echoed in the instruct-written
        # prefix; drop it at CPU cost with a named reason instead.
        if story.count(answer) != 1:
            return None, "answer_not_unique_in_story"
        # ...and the answer-multiplicity axis is only ONE of the gate's verdicts.
        # Run the CONSUMER'S OWN GATE on the assembled story and drop on any
        # rejection, so no gate class can reach the capture's assert. Measured
        # residual this closes: 3/2089 rows whose answer ENDS with
        # attribution-shaped words ("...as the Assistant explained,") — the
        # closing quote appended two lines up then supplies the quote character
        # the attribution regex needs, so the reassembled story carries a SECOND
        # attribution match and the gate returns `attribution_multi`. The answer
        # text alone carries zero, so this is a product of the reassembly, and it
        # is invisible to any check that does not run the real gate.
        v1_turn, v1_reason = _v1_gate_cached()(story, answer)
        if v1_reason != "ok" or v1_turn is None:
            return None, f"v1_gate_{v1_reason}"
        # The gate returning "ok" is only the FIRST of the capture's two
        # trust-boundary checks. The second compares the STORED spans against the
        # gate's re-derivation key-by-key, and a row can pass the gate while
        # disagreeing on a span — exactly the a_start +1 class the leading-space
        # lstrip above fixes. Run that comparison HERE too, so any future
        # divergence is a gen-time drop rather than a mid-capture assert.
        for span_key in ("q_start", "q_end", "boundary_end", "a_start", "a_end"):
            if span_key in v1_turn and span_key in turn and v1_turn[span_key] != turn[span_key]:
                return None, f"v1_gate_span_mismatch_{span_key}"
        return {
            "conv_id": pool_row["conv_id"],
            "story": story,
            "answer": answer,
            "parsed_turns": [turn],
            "shape": shape,
            "model": model_key,
            "provenance": c.PROV_ONPOLICY,
            "trailing_quote_stripped": stripped,
            "leading_ws_stripped": lead_ws,
            "prefix_chars": a_start,
        }, "ok"

    answer = answer_text.strip()
    if len(answer) < ANSWER_CHAR_MIN:
        return None, "answer_too_short"
    # Comparator-convs schema: `to_single_turn` maps prompt -> u1, response -> a1.
    return {
        "conv_id": pool_row["conv_id"],
        "prompt": pool_row["question"],
        "response": answer,
        "shape": shape,
        "model": model_key,
        "provenance": c.PROV_ONPOLICY,
    }, "ok"


def validate_row(row: dict, tokenizer, *, shape: str) -> bool:
    """Render the row through the CONSUMER's OWN function (the #825 gen-time gate).

    Not a mirrored re-implementation: a drift between a local copy of the span
    arithmetic and the capture's real render is exactly the class this guards.
    """
    import issue1345_boundary_ablation_capture as cap

    if shape == SHAPE_STORY_SLOT:
        # One arg per line (magic trailing comma keeps it exploded): packed onto a
        # single line the identifier chain trips gitleaks' generic-api-key entropy
        # rule, and a line-number-keyed .gitleaksignore waiver drifts on any edit
        # above it.
        rendered = cap.render_boundary_turn(
            row["story"],
            row["parsed_turns"][0],
            row["conv_id"],
            tokenizer,
            arm=cap.V1_ARM,
        )
    else:
        conv = {"conv_id": row["conv_id"], "u1": row["prompt"], "a1": row["response"]}
        rendered = cap.render_comparator_turn(conv, tokenizer, comparator=SHAPE_COMPARATOR[shape])
    return rendered is not None


# ---------------------------------------------------------------------------
# Generation (chunked, per-chunk checkpoint, conv_id-keyed resume)
# ---------------------------------------------------------------------------
def generate_answers(
    pool: list[dict], out_path: Path, fp: str, tokenizer, llm, *, shape: str, model_key: str
) -> list[dict]:
    """One answer per pool row; per-chunk JSONL checkpoint keyed on conv_id."""
    from vllm import SamplingParams

    meta_path = out_path.with_suffix(".meta.json")
    done_ids: set[str] = set()
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            done_ids = {r["conv_id"] for r in c.read_jsonl(out_path)}
            print(f"[gen] resume: {len(done_ids)} answers already on disk", flush=True)
        else:
            raise RuntimeError(
                f"{out_path} exists with a DIFFERENT generation fingerprint "
                f"({meta.get('fingerprint')} != {fp}) — refusing to mix recipes; "
                "move the stale file aside"
            )
    else:
        c.write_json(meta_path, {"fingerprint": fp, "n_rows": len(pool), "shape": shape})

    todo = [r for r in pool if r["conv_id"] not in done_ids]
    sampling = SamplingParams(
        temperature=c.STORY_TEMPERATURE,
        max_tokens=ONPOLICY_MAX_NEW_TOKENS,
        stop=SHAPE_STOP[shape],
        seed=None,
    )
    n_chunks = (len(todo) + g.VLLM_CHUNK_SIZE - 1) // g.VLLM_CHUNK_SIZE
    for ci in range(0, len(todo), g.VLLM_CHUNK_SIZE):
        chunk = todo[ci : ci + g.VLLM_CHUNK_SIZE]
        prompts = [build_gen_prompt(r, shape=shape) for r in chunk]
        print(
            f"[vllm-chunk] {shape} gen chunk {ci // g.VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts, model={model_key})",
            flush=True,
        )
        outs = llm.generate(prompts, sampling, use_tqdm=False)
        new_rows = []
        for r, o in zip(chunk, outs, strict=True):
            new_rows.append(
                {
                    "conv_id": r["conv_id"],
                    "shape": shape,
                    "model": model_key,
                    "answer_text": o.outputs[0].text,
                    "finish_reason": o.outputs[0].finish_reason,
                }
            )
        c.append_jsonl(out_path, new_rows)
    return c.read_jsonl(out_path) if out_path.exists() else []


def keep_rows(
    raw_rows: list[dict], pool: list[dict], tokenizer, *, shape: str, model_key: str
) -> tuple[list[dict], dict]:
    """Assemble + gen-time-validate every generated answer; count every drop."""
    by_id = {r["conv_id"]: r for r in pool}
    counts = {
        "raw": len(raw_rows),
        "kept": 0,
        "answer_too_short": 0,
        "answer_not_unique_in_story": 0,
        "render_none": 0,
        "finish_length_capped": 0,
        "trailing_quote_stripped": 0,
        "leading_ws_stripped": 0,
        "v1_gate_span_mismatch": 0,
    }
    kept: list[dict] = []
    for raw in raw_rows:
        pool_row = by_id.get(raw["conv_id"])
        assert pool_row is not None, (
            f"raw answer for {raw['conv_id']} is not in the pool — stale raw file vs a new "
            "fingerprint (the resume guard should have refused this)"
        )
        if raw.get("finish_reason") == "length":
            counts["finish_length_capped"] += 1
        row, reason = assemble_row(pool_row, raw["answer_text"], shape=shape, model_key=model_key)
        if row is None:
            # The consumer's gate owns its own reason vocabulary, so a
            # `v1_gate_*` verdict is registered as it appears — a gate reason we
            # have not seen before still gets COUNTED rather than crashing the
            # run or vanishing. Every other reason must be pre-declared, so the
            # assert still catches a typo'd or unaccounted local drop class.
            if reason.startswith("v1_gate_span_mismatch_"):
                counts["v1_gate_span_mismatch"] += 1
                continue
            if reason.startswith("v1_gate_"):
                counts.setdefault(reason, 0)
            assert reason in counts, f"unaccounted drop reason {reason!r}"
            counts[reason] += 1
            continue
        if not validate_row(row, tokenizer, shape=shape):
            counts["render_none"] += 1
            continue
        if row.get("trailing_quote_stripped"):
            counts["trailing_quote_stripped"] += 1
        if row.get("leading_ws_stripped"):
            counts["leading_ws_stripped"] += 1
        # Y_BOUNDARY sensitivity split (#1345 fits): a cap-truncated answer ends
        # MID-SENTENCE, so the boundary target read just after it is an artifact
        # of the cap rather than a natural end-of-answer transition. Y_MEAN over
        # the answer span is unaffected. Carried per row (not just aggregated in
        # the yield report) so the fits can split the boundary target instead of
        # pooling two different objects.
        row["finish_reason"] = raw.get("finish_reason")
        row["capped"] = raw.get("finish_reason") == "length"
        kept.append(row)
        counts["kept"] += 1
    print(f"[keep] {shape} ({model_key}): {counts}", flush=True)
    return kept, counts


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def stem_for(shape: str, model_key: str) -> str:
    return f"{SHAPE_SLUG[shape]}_{model_key}"


def raw_path(out_dir: Path, shape: str, model_key: str) -> Path:
    return out_dir / f"raw_onpolicy_{stem_for(shape, model_key)}.jsonl"


def rows_path(out_dir: Path, shape: str, model_key: str) -> Path:
    return out_dir / f"onpolicy_rows_{stem_for(shape, model_key)}.jsonl"


def yield_path(out_dir: Path, shape: str, model_key: str) -> Path:
    return out_dir / f"onpolicy_yield_{stem_for(shape, model_key)}.json"


def manifest_path(out_dir: Path, shape: str, model_key: str) -> Path:
    return out_dir / f"onpolicy_manifest_{stem_for(shape, model_key)}.json"


def bundle_files(out_dir: Path, shape: str, model_key: str) -> list[str]:
    names = [
        raw_path(out_dir, shape, model_key).name,
        raw_path(out_dir, shape, model_key).with_suffix(".meta.json").name,
        rows_path(out_dir, shape, model_key).name,
        yield_path(out_dir, shape, model_key).name,
    ]
    return [n for n in names if (out_dir / n).exists()]


def write_yield_report(out_dir: Path, shape: str, model_key: str, fp: str, payload: dict) -> None:
    """Yield + drop accounting, plus the realized env the capture guard reads."""
    report = {
        "metadata": c.metadata(
            c.GEN_SEED, int(payload.get("n_kept", 0)), "scripts/issue1345_onpolicy_answers_gen.py"
        ),
        "variant": c.VARIANT,
        "shape": shape,
        "model": model_key,
        "provenance": c.PROV_ONPOLICY,
        "consumer": SHAPE_CONSUMER[shape],
        "bundle_fingerprint": fp,
        "story_character_name": c.STORY_CHARACTER_NAME,
        "max_new_tokens": ONPOLICY_MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "prompt_token_budget": PROMPT_TOKEN_BUDGET,
        "stop_strings": SHAPE_STOP[shape],
        "answer_char_min": ANSWER_CHAR_MIN,
        **payload,
    }
    c.write_json(yield_path(out_dir, shape, model_key), report)


def persist_bundle(out_dir: Path, shape: str, model_key: str, fp: str, smoke: bool) -> None:
    """Upload the rollout text + rows + yield report to HF NOW (upload-by-default)."""
    assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — cannot persist the on-policy bundle"
    files = bundle_files(out_dir, shape, model_key)
    assert rows_path(out_dir, shape, model_key).name in files, files
    manifest = {
        "metadata": c.metadata(c.GEN_SEED, len(files), "scripts/issue1345_onpolicy_answers_gen.py"),
        "variant": c.VARIANT,
        "shape": shape,
        "model": model_key,
        "provenance": c.PROV_ONPOLICY,
        "consumer": SHAPE_CONSUMER[shape],
        "bundle_fingerprint": fp,
        "files": files,
    }
    c.write_json(manifest_path(out_dir, shape, model_key), manifest)
    prefix = g._stories_hf_prefix(smoke)
    g._hf_upload_folder(
        out_dir,
        prefix,
        [
            f"*onpolicy*{stem_for(shape, model_key)}*",
            f"*raw_onpolicy_{stem_for(shape, model_key)}*",
        ],
        f"issue-1345: on-policy {shape} answers ({model_key}, fp {fp})",
    )
    print(f"[gen] persisted {shape} on-policy answers -> {prefix} (fp {fp})", flush=True)


# ---------------------------------------------------------------------------
# Import check (every deferred import on the real code path)
# ---------------------------------------------------------------------------
def _import_check() -> None:
    """Resolve every deferred import this script reaches in production."""
    import issue1345_boundary_ablation_capture as cap  # noqa: F401

    from huggingface_hub import upload_folder  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401
    from vllm import LLM, SamplingParams  # noqa: F401

    from explore_persona_space.experiments.issue_825.common import (  # noqa: F401
        MODEL_INSTRUCT,
        MODEL_PRETRAINED,
    )
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        assert_hub_dir_filecounts,
        retry_transient,
    )

    print("import-ok:", SHAPES, sorted(ONPOLICY_ANSWER_VARIANTS), flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shape", choices=SHAPES, help="which on-policy answer shape to generate")
    ap.add_argument("--model", choices=c.MODELS, help="the model that WRITES the answers")
    ap.add_argument("--out-dir", type=Path, default=c.STORIES_DIR)
    ap.add_argument("--dl-dir", type=Path, default=c.PARENT_DL_DIR)
    ap.add_argument("--matched-dir", type=Path, default=c.MATCHED_DIR)
    ap.add_argument(
        "--stories-jsonl",
        type=Path,
        default=None,
        help="story_slot ONLY: kept-stories JSONL to slice prefixes from "
        "(default: the V1 anchor's sha-pinned parent bundle)",
    )
    ap.add_argument("--n-rows", type=int, default=0, help="0 = the whole filtered pool")
    ap.add_argument(
        "--yield-floor",
        type=int,
        default=0,
        help="0 = 80%% of the filtered pool (the on-policy-completions floor)",
    )
    ap.add_argument("--skip-upload", action="store_true", help="local-only (smoke plumbing)")
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_N_ROWS} rows, floor 1")
    ap.add_argument(
        "--verify-pool",
        action="store_true",
        help="zero-GPU CPU preflight: pool + prompt budgets + fingerprint, then exit 0",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import on the real code path and exit 0",
    )
    args = ap.parse_args()

    if args.import_check:
        _import_check()
        return

    assert args.shape and args.model, "--shape and --model are required"
    assert_onpolicy_env(args.shape, args.model)
    if args.stories_jsonl is not None:
        assert args.shape == SHAPE_STORY_SLOT, "--stories-jsonl applies to --shape story_slot only"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_825.common import MODEL_INSTRUCT, MODEL_PRETRAINED

    model_id = MODEL_INSTRUCT if args.model == "instruct" else MODEL_PRETRAINED
    # Prompts are built from the CAPTURE's render, which the capture tokenizes
    # with the INSTRUCT tokenizer — so the budget filter must measure with the
    # same one regardless of which model writes the answers.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)

    if args.shape == SHAPE_STORY_SLOT:
        pool, pool_counts = load_story_slot_pool(args.dl_dir, args.stories_jsonl)
    else:
        pool, pool_counts = load_comparator_pool(args.matched_dir, args.dl_dir)
    pool, budget_counts = filter_pool_by_prompt_budget(pool, tokenizer, shape=args.shape)

    if args.smoke:
        pool = pool[:SMOKE_N_ROWS]
        print(f"[smoke] limiting to {len(pool)} {args.shape} rows", flush=True)
    elif args.n_rows:
        pool = pool[: args.n_rows]
        print(f"[gen] --n-rows: limiting to {len(pool)} rows", flush=True)

    fp = bundle_fingerprint(args.shape, args.model, [r["conv_id"] for r in pool])
    floor = args.yield_floor or int(0.8 * len(pool))
    yield_floor = g.resolve_yield_floor(args.smoke, floor)
    print(
        f"[gen] shape={args.shape} model={args.model} ({model_id}) variant={c.VARIANT} "
        f"n_pool={len(pool)} fp={fp} yield_floor={yield_floor}",
        flush=True,
    )

    if args.verify_pool:
        # Zero-GPU preflight: prove the pool resolves, the prompts fit, and one
        # row renders through the consumer's own function before any pod boots.
        probe, probe_reason = assemble_row(
            pool[0],
            pool[0].get("source_answer", "x" * (ANSWER_CHAR_MIN + 8)),
            shape=args.shape,
            model_key=args.model,
        )
        assert probe is not None, f"probe row assembled to None ({probe_reason})"
        assert validate_row(probe, tokenizer, shape=args.shape), (
            "probe row failed the consumer's own render — span/format bug before any GPU spend"
        )
        print(
            f"[verify-pool] OK shape={args.shape} model={args.model} n_pool={len(pool)} fp={fp} "
            f"pool_counts={pool_counts} budget_counts={budget_counts} "
            f"probe_conv_id={probe['conv_id']}",
            flush=True,
        )
        sys.exit(0)

    from vllm import LLM

    llm = LLM(
        model=model_id,
        seed=c.GEN_SEED,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        # Live-probed, NOT hardcoded: a GPU-SHARED fellows node would otherwise
        # crash at EngineCore init (#1902). This round runs on exclusive hosts,
        # so the exclusive-host cap applies (0.85, not the shared-node 0.55).
        gpu_memory_utilization=resolve_vllm_util(cap=EXCLUSIVE_HOST_UTIL_CAP),
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=(
            False if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1" else None
        ),
    )

    raw = generate_answers(
        pool,
        raw_path(out_dir, args.shape, args.model),
        fp,
        tokenizer,
        llm,
        shape=args.shape,
        model_key=args.model,
    )
    kept, keep_counts = keep_rows(raw, pool, tokenizer, shape=args.shape, model_key=args.model)

    rp = rows_path(out_dir, args.shape, args.model)
    if rp.exists():
        rp.unlink()
    c.append_jsonl(rp, kept)
    write_yield_report(
        out_dir,
        args.shape,
        args.model,
        fp,
        {
            "n_pool": len(pool),
            "n_kept": len(kept),
            "yield_floor": yield_floor,
            "pool_counts": pool_counts,
            "budget_counts": budget_counts,
            "keep_counts": keep_counts,
            "rows_file": rp.name,
        },
    )

    if args.skip_upload:
        print(f"[gen] --skip-upload: {len(kept)} rows -> {rp}, LOCAL ONLY", flush=True)
    else:
        persist_bundle(out_dir, args.shape, args.model, fp, args.smoke)

    g.enforce_yield_floor(len(kept), yield_floor)
    print(
        f"[done] {args.shape} ({args.model}): kept={len(kept)}/{len(pool)} -> {rp}",
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
