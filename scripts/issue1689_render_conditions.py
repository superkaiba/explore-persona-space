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
        # abstract shape here.
        row["prompt_source"] = "chat_template"
    # 2) naturalistic — plain text with role headers `User:` / `Assistant:` (or
    # persona-labeled) per plan §4.
    elif condition.framing == "naturalistic":
        assistant_tag = identity_display(condition)
        user_tag = "User"  # user arm is also spoken as `User` in the transcript
        # For character arms, u2 is spoken BY THE USER (the character SPEAKS
        # a2, not u2). For the user arm, u2 IS what the model produces.
        # Naturalistic user-in-frame: {User_prev: u1} \n\n {Assistant_prev: a1}
        # \n\n {User: u2} \n\n {Assistant_next:}
        if condition.is_user:
            # The user-arm asks the MODEL to produce u2 (the user's next turn),
            # so the FILLED "u2" is what the model just wrote (or an lmsys/haiku
            # substitute).
            text = (
                f"{user_tag}: {u1}\n\n"
                f"{assistant_tag}: {a1}\n\n"
                f"{user_tag}: "  # a2 slot: MODEL generates the user's turn 2
            )
            # For the user arm's naturalistic framing, u2_text is the DV: the
            # model generates it. We only render up to the a2 slot.
            row["prompt_source"] = "naturalistic_user_arm"
        else:
            text = (
                f"{user_tag}: {u1}\n\n"
                f"{assistant_tag}: {a1}\n\n"
                f"{user_tag}: {u2_text}\n\n"
                f"{assistant_tag}: "  # a2 slot
            )
            row["prompt_source"] = "naturalistic_assistant"
        row["prompt_text"] = text
    # 3) story framing — narrative prose w/ dialogue.
    elif condition.framing == "story":
        if condition.identity == "assistant":
            template = _STORY_ASSISTANT_TEMPLATE
            text = template.format(u1=u1, a1=a1, u2=u2_text)
        elif condition.is_user:
            # Novel construction: user-in-story - the story has u2 as content
            # the MODEL generates (frames the model as writing the user's line).
            template = _STORY_USER_TEMPLATE
            text = template.format(u1=u1, a1=a1, u2=u2_text)
        else:  # character
            name = identity_display(condition)
            desc = PERSONAS[name]
            template = _STORY_CHARACTER_TEMPLATE
            text = template.format(name=name, desc=desc, u1=u1, a1=a1, u2=u2_text)
        row["prompt_text"] = text
        row["prompt_source"] = "story"
    else:
        raise ValueError(f"unknown framing: {condition.framing}")

    return row


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
                fout.write(json.dumps(row) + "\n")
                n_rendered += 1
        print(f"[render] {cond.slug} -> {out_path} ({len(rows)} rows)", flush=True)

    print(f"[render] done: {n_rendered} total rows across {len(conditions)} conditions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
