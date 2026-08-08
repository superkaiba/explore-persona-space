"""Issue #1689 Phase A — render each conversation into 21 conditions.

Extends #825's `render_chat` / `render_naturalistic` to characters (Wren /
HELIOS / Dana) and to the user-in-story construction the plan introduces
(never attempted in any prior task). See plan §4 "Framing conventions".

For each conv_id in the input corpus, emits 21 rendered rows (one per
condition), keyed by (conv_id, condition_slug). The u2 slot content is set
by the user-provenance arm:
  - lmsys: use the LMSYS turn-3 text if present, else placeholder + drop flag.
  - haiku: to be filled by `issue1689_haiku_u2_gen.py` post-render (this
    script emits a `u2_missing_provenance="haiku"` sentinel that phase reads).
  - onpolicy: to be filled by `issue1689_gen_onpolicy.py` post-render.

For chat template: injects the character description as the system prompt.
For naturalistic: renders as `User: … / Assistant: …` (or `Wren:`, `HELIOS:`,
`Dana:` for character conditions) plain text.
For story: uses a narrator preamble ("A user named Alex approached HELIOS…")
and dialogue tags.

Row schema (JSONL):
  {conv_id, condition, model, prompt_text, u1, a1, u2_source, u2_text?,
   assistant_label, system_prompt?, framing}

`prompt_text` is the TEXT the model is fed up to but NOT INCLUDING a2.
Capture is teacher-forced on `prompt_text + a2` (Phase C).

Smoke: `--in /tmp/i1689-smoke/two_turn.jsonl --conditions all --out /tmp/i1689-smoke/rendered/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    assert (repo_root / "scripts" / "issue1689_common.py").exists(), repo_root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_syspath()

from scripts.issue1689_common import (  # noqa: E402
    CONDITION_TABLE,
    Condition,
    PERSONAS,
    SLUG_TO_CONDITION,
    identity_display,
    system_prompt_for,
)

# --- Story framing preamble templates -------------------------------------

_STORY_ASSISTANT_TEMPLATE = (
    "A person sat down at their computer and asked an AI assistant a question. "
    'The user typed: "{u1}". The assistant thought for a moment, then wrote back: '
    '"{a1}". Later, the user typed a follow-up: "{u2}". The assistant paused, '
    "considered the question, and responded: "
)

_STORY_USER_TEMPLATE = (
    "A user named Alex sat down at their computer to talk with an AI assistant. "
    'Alex typed: "{u1}". The assistant replied: "{a1}". Alex thought about the '
    'response, then typed a follow-up: "{u2}". The assistant wrote back: '
)

_STORY_CHARACTER_TEMPLATE = (
    'A person came to talk to {name}, {desc}. They said: "{u1}". {name} replied: '
    '"{a1}". They then said: "{u2}". {name} considered the question and said: '
)


def render_condition(
    conv: dict,
    condition: Condition,
    *,
    u2_text: str | None = None,
) -> dict:
    """Render one (conv, condition) row.

    ``u2_text`` = the text of the follow-up user turn for THIS condition. For
    the lmsys provenance we read it from conv["u2_lmsys"] (populated by the
    corpus gen at 2-turn filter; the LMSYS-3rd-turn if present else a
    placeholder). For the haiku/onpolicy arms this script emits a placeholder
    sentinel — later phases fill it in and re-render.
    """
    u1 = conv["u1"]
    a1 = conv["a1"]
    if u2_text is None:
        u2_text = conv.get("u2_lmsys", "")

    row: dict = {
        "conv_id": conv["conv_id"],
        "condition": condition.slug,
        "framing": condition.framing,
        "identity": condition.identity,
        "provenance": condition.provenance,
        "u1": u1,
        "a1": a1,
        "u2_text": u2_text,
        "system_prompt": system_prompt_for(condition),
        "assistant_label": identity_display(condition),
    }

    # For the distinct prefix / context / answer arms per plan §4 + CLAUDE.md
    # "Prefix mapping AND context mapping", the renderer emits THREE text
    # segments (never re-joined at capture time — capture concatenates in the
    # SAME order the render used to compute character offsets):
    #   prefix_text_only  = everything BEFORE u2 (system + u1 + a1 + role hdr)
    #   u2_text_marked    = the u2 segment itself (may be empty for user-arm
    #                       where u2 IS the DV; character-offset-continuous
    #                       with prefix_text_only)
    #   context_tail      = interstitial between u2 and a2 (e.g. "\n\nAssistant: ")
    # The capture rig tokenizes  prefix_text_only + u2_text_marked + context_tail
    # + a2_text  in ONE pass with return_offsets_mapping=True and reads:
    #   prefix_end_char_offset  = len(prefix_text_only)
    #   context_end_char_offset = len(prefix_text_only + u2_text_marked + context_tail)
    # then maps each char offset → token index via offset_mapping (BPE-safe;
    # avoids the plain-text-boundary merge trap in gotchas.md § "Plain-text
    # span boundaries are the WORST case").
    #
    # Also retained: prompt_text (= prefix_text_only + u2_text_marked +
    # context_tail — the exact model input up to a2), for backwards compat
    # with any downstream reader.

    # 1) chat template - inject system prompt when character; user/assistant
    # arms use the base chat template.
    if condition.framing == "chat":
        messages: list[dict] = []
        sysp = system_prompt_for(condition)
        if sysp is not None:
            messages.append({"role": "system", "content": sysp})
        messages.append({"role": "user", "content": u1})
        messages.append({"role": "assistant", "content": a1})
        messages.append({"role": "user", "content": u2_text})
        row["messages"] = messages
        # For a Qwen chat-template render we defer to the tokenizer at capture
        # time (`apply_chat_template(add_generation_prompt=True)`). Record the
        # abstract shape here. Capture also splits at the u2 message boundary
        # using apply_chat_template on the leading conversation prefix (up to
        # a1) vs the full conversation (through u2) — see capture_cell's
        # _chat_offsets helper.
        row["prompt_source"] = "chat_template"
        # For chat framing, prefix_text_only + u2 + context_tail are rendered
        # at capture time via apply_chat_template. Renderer stores the
        # per-role parts so the capture rig can reconstruct all three.
        row["prefix_text_only"] = None  # capture reconstructs via chat_template
        row["u2_text_marked"] = u2_text
        row["context_tail"] = None
    # 2) naturalistic — plain text with role headers `User:` / `Assistant:` (or
    # persona-labeled) per plan §4.
    elif condition.framing == "naturalistic":
        assistant_tag = identity_display(condition)
        user_tag = "User"  # user arm is also spoken as `User` in the transcript
        # For character arms, u2 is spoken BY THE USER (the character SPEAKS
        # a2, not u2). For the user arm, u2 IS what the model produces.
        # Naturalistic user-in-frame: {User_prev: u1} \n\n {Assistant_prev: a1}
        # \n\n {User: u2} \n\n {Assistant_next:}
        prefix_only = f"{user_tag}: {u1}\n\n{assistant_tag}: {a1}\n\n{user_tag}: "
        if condition.is_user:
            # The user-arm asks the MODEL to produce u2 (the user's next turn),
            # so the FILLED "u2" is what the model just wrote (or an lmsys/haiku
            # substitute). context_end == prefix_end (no u2 in the input prompt
            # for THIS arm's on-policy read — capture treats the two arms as
            # identical here, per plan §4 "user-in-frame u2 IS the DV").
            u2_marked = ""
            context_tail = ""
            row["prompt_source"] = "naturalistic_user_arm"
        else:
            u2_marked = u2_text
            context_tail = f"\n\n{assistant_tag}: "
            row["prompt_source"] = "naturalistic_assistant"
        row["prefix_text_only"] = prefix_only
        row["u2_text_marked"] = u2_marked
        row["context_tail"] = context_tail
        row["prompt_text"] = prefix_only + u2_marked + context_tail
    # 3) story framing — narrative prose w/ dialogue.
    elif condition.framing == "story":
        # Story templates embed u2 as a QUOTED DIALOGUE turn: `"..."`. The
        # boundary layout is (prefix_up_to_open_quote) + u2 + (close_quote +
        # narrative_tail). To keep char offsets non-ambiguous, we split each
        # template on the {u2} placeholder BEFORE formatting — the head
        # holds everything up to (but not including) the open quote around
        # u2's content, and the tail holds the closing quote + narrative
        # bridge + a2-slot lead-in.
        if condition.identity == "assistant":
            template = _STORY_ASSISTANT_TEMPLATE
            fmt_kwargs = {"u1": u1, "a1": a1, "u2": u2_text}
        elif condition.is_user:
            template = _STORY_USER_TEMPLATE
            fmt_kwargs = {"u1": u1, "a1": a1, "u2": u2_text}
        else:  # character
            name = identity_display(condition)
            desc = PERSONAS[name]
            template = _STORY_CHARACTER_TEMPLATE
            fmt_kwargs = {"name": name, "desc": desc, "u1": u1, "a1": a1, "u2": u2_text}
        # Split template around the {u2} placeholder so prefix_text_only holds
        # everything (verbatim) up to the u2 content, and context_tail holds
        # everything after.
        head_tpl, tail_tpl = template.split("{u2}", 1)
        # Strip u2 from fmt_kwargs when formatting head and tail (they no
        # longer contain {u2} but may contain other placeholders).
        head_kwargs = {k: v for k, v in fmt_kwargs.items() if k != "u2"}
        prefix_only = head_tpl.format(**head_kwargs)
        u2_marked = u2_text
        context_tail = tail_tpl.format(**head_kwargs)
        row["prefix_text_only"] = prefix_only
        row["u2_text_marked"] = u2_marked
        row["context_tail"] = context_tail
        row["prompt_text"] = prefix_only + u2_marked + context_tail
        row["prompt_source"] = "story"
    else:
        raise ValueError(f"unknown framing: {condition.framing}")

    return row


# Structural render-side validation. A rendered row is VALID for downstream
# vLLM consumption iff at least one of these holds:
#   - chat framing: `messages` present, non-empty, and each message has
#     non-empty content.
#   - naturalistic / story: `prompt_text` present and non-empty.
# A row failing this is a broken render — Phase B / Phase C both crash on it
# (vLLM raises `ValueError: The decoder prompt cannot be empty`), so we catch
# it here at write time instead of one full pod cycle later.
def validate_rendered_row(row: dict) -> tuple[bool, str]:
    """Return (ok, reason). ok=True iff the row is downstream-usable."""
    framing = row.get("framing")
    if framing == "chat":
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return False, "chat row missing/empty messages"
        for m in msgs:
            if not isinstance(m, dict):
                return False, "chat row: non-dict message"
            content = m.get("content", "")
            if content is None or (isinstance(content, str) and content == ""):
                # Note: an EMPTY user turn (u2="") is legal for the user-arm
                # in naturalistic/story but NOT in chat (vLLM chokes on empty
                # message content in the chat template render).
                return False, f"chat row: empty content on role={m.get('role')!r}"
        return True, "ok"
    # naturalistic / story: rely on prompt_text
    prompt_text = row.get("prompt_text", "")
    if not prompt_text or not prompt_text.strip():
        return False, f"{framing} row: empty prompt_text"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--conditions",
        default="all",
        help='Either "all" or a comma-separated list of condition slugs.',
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--strict-validate",
        action="store_true",
        help=(
            "Fail loud if ANY rendered row fails structural validation "
            "(empty prompt_text on naturalistic/story, empty message "
            "content on chat). Recommended for production; the smoke path "
            "should stay lenient so a placeholder u2 does not abort."
        ),
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.conditions == "all":
        conditions = list(CONDITION_TABLE)
    else:
        wanted = args.conditions.split(",")
        conditions = [SLUG_TO_CONDITION[s.strip()] for s in wanted if s.strip()]

    with args.in_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(
        f"[render] loaded {len(rows)} conversations; rendering {len(conditions)} conditions",
        flush=True,
    )

    # For the user-arm rows, we don't have haiku/onpolicy u2 yet; those are
    # filled in by their respective phases. For the assistant/character arms
    # we need u2 (a real second-user turn) — the smoke corpus doesn't have
    # LMSYS turn-3, so we fall back to a placeholder to preserve the row.
    n_rendered = 0
    validation_failures: list[dict] = []
    for cond in conditions:
        out_path = args.out / f"{cond.slug}.jsonl"
        with out_path.open("w") as fout:
            for conv in rows:
                # Determine u2 text per provenance
                if cond.is_user and cond.provenance != "lmsys":
                    # placeholder — will be filled by later phase
                    u2_text = "<UNFILLED_U2_PLACEHOLDER>"
                else:
                    u2_text = conv.get("u2_lmsys", "Can you say a bit more about that?")
                row = render_condition(conv, cond, u2_text=u2_text)
                ok, reason = validate_rendered_row(row)
                if not ok:
                    validation_failures.append(
                        {
                            "conv_id": row.get("conv_id"),
                            "condition": cond.slug,
                            "framing": cond.framing,
                            "reason": reason,
                        }
                    )
                fout.write(json.dumps(row) + "\n")
                n_rendered += 1
        print(f"[render] {cond.slug} -> {out_path} ({len(rows)} rows)", flush=True)

    if validation_failures:
        n_fail = len(validation_failures)
        print(
            f"[render] WARNING: {n_fail} rows failed structural validation "
            f"(sample: {validation_failures[:3]!r})",
            flush=True,
        )
        if args.strict_validate:
            print(
                f"[render] --strict-validate: FAILING because {n_fail} rows "
                f"would crash vLLM at the decoder-prompt-empty check.",
                flush=True,
            )
            return 2

    print(f"[render] done: {n_rendered} total rows across {len(conditions)} conditions")
    return 0


if __name__ == "__main__":
    import os

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGBART pointer. main()'s writes are
    # already flushed via explicit fh.close(); atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
