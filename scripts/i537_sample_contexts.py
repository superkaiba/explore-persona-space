"""Issue #537 P0 -- deterministic WildChat + PersonaHub context sampling.

Plan v6 §4.0 procedures (this script is their single executable form):

WildChat (``allenai/WildChat-1M``): filter ``language == "English"``,
``toxic == False``, ``redacted == False``; dedupe by ``conversation_hash``;
token bins (Qwen tokens of the chat-templated prefix): short = first exchange
150-500; long = first 4 exchanges ≤ 2,000; xlong = first ~8 exchanges
truncated at exchange boundaries into 4,000-5,000; xxlong = 7,000-9,000
(exchange floor ≥ 8, relaxed to ≥ 4 on an empty slot BEFORE touching the
token bin). Reject conversations containing ``※``, "courthouse", "Ridgway",
or Haiku-flagged unsafe; topic via one Haiku call per candidate; shuffle seed
537; first passer per (bin, topic) slot. ``prefix_token_len`` recorded per
instance.

PersonaHub (``proj-persona/PersonaHub`` config ``persona``): English, 1-3
sentences, 15-80 Qwen tokens, Haiku realism/professional screen, no
occupational overlap with the 4 house personas, no safety-adjacent
occupations; shuffle seed 537; first 4 passers → ``sp_ph1``, ``sp_ph2``,
``sp_ph3_ho``, ``neg_sp_ph4``.

Smoke mode: ``--max-rows N`` bounds the stream scan; ``--skip-screens`` skips
the Haiku screens (verdicts recorded as ``"skipped-smoke"``);
``--allow-partial`` exits 0 with unfilled slots listed (smoke ONLY -- the real
P0 run fails loud on any empty slot, G0(ii)).

Usage:
    uv run python scripts/i537_sample_contexts.py \
        --out data/issue_537/contexts/sampled_contexts.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_sample_contexts")

SEED = 537
HAIKU_MODEL = "claude-haiku-4-5-20251001"
QWEN_ID = "Qwen/Qwen2.5-7B-Instruct"
BANNED_SUBSTRINGS = ("※", "courthouse", "Courthouse", "Ridgway")

HOUSE_OCCUPATIONS = (
    "software engineer",
    "medical doctor",
    "kindergarten teacher",
    "police officer",
)

# (cid, bin, topic_slot). Topic slots are matched against the Haiku topic label.
WILDCHAT_SLOTS: list[tuple[str, str, str]] = [
    ("wc_short_code", "short", "coding"),
    ("wc_short_advice", "short", "advice"),
    ("wc_long_write", "long", "writing"),
    ("wc_short_ho", "short", "travel-or-cooking"),
    ("wc_long_ho", "long", "other"),
    ("wc_xlong_ho", "xlong", "any"),
    ("wc_xxlong_ho", "xxlong", "any-distinct"),
    ("neg_wc_short", "short", "tech-support"),
]

TOKEN_BINS = {  # (lo, hi) inclusive token bounds per bin
    "short": (150, 500),
    "long": (0, 2000),
    "xlong": (4000, 5000),
    "xxlong": (7000, 9000),
}
EXCHANGES = {"short": 1, "long": 4, "xlong": 8, "xxlong": 8}  # xlong/xxlong = floor


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _haiku_json(client, prompt: str) -> dict:
    """One Haiku call returning a parsed JSON object (fail loud on non-JSON)."""
    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m is None:
        raise ValueError(f"Haiku screen returned non-JSON: {text[:200]!r}")
    return json.loads(m.group(0))


def _prefix_token_len(tokenizer, messages: list[dict]) -> int:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def _contains_banned(text: str) -> bool:
    return any(b in text for b in BANNED_SUBSTRINGS)


def _conversation_to_exchanges(conv: list[dict]) -> list[tuple[str, str]]:
    """Pair the conversation into (user, assistant) exchanges; drop a dangling turn."""
    pairs = []
    i = 0
    while i + 1 < len(conv):
        a, b = conv[i], conv[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            pairs.append((a["content"], b["content"]))
            i += 2
        else:
            i += 1
    return pairs


def _truncate_to_bin(
    tokenizer, exchanges: list[tuple[str, str]], lo: int, hi: int, floor: int
) -> tuple[list[dict], int] | None:
    """Largest exchange-boundary prefix with token length in [lo, hi] and ≥ floor exchanges."""
    best = None
    msgs: list[dict] = []
    for k, (u, a) in enumerate(exchanges, start=1):
        msgs = [*msgs, {"role": "user", "content": u}, {"role": "assistant", "content": a}]
        tl = _prefix_token_len(tokenizer, msgs)
        if tl > hi:
            break
        if tl >= lo and k >= floor:
            best = (list(msgs), tl)
    return best


def sample_wildchat(  # noqa: C901 - the slot-matching state machine reads clearest flat
    tokenizer,
    client,
    *,
    max_rows: int | None,
    skip_screens: bool,
    exchange_floor_xxlong: int = 8,
) -> dict[str, dict]:
    """First-passer-per-slot deterministic WildChat sampling (plan §4.0)."""
    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    filled: dict[str, dict] = {}
    seen_hashes: set[str] = set()
    used_topics_long: set[str] = set()
    n_scanned = 0
    for row in ds:
        if max_rows is not None and n_scanned >= max_rows:
            break
        n_scanned += 1
        if len(filled) == len(WILDCHAT_SLOTS):
            break
        if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
            continue
        chash = row.get("conversation_hash") or ""
        if chash in seen_hashes:
            continue
        seen_hashes.add(chash)
        exchanges = _conversation_to_exchanges(row.get("conversation") or [])
        if not exchanges:
            continue
        full_text = "\n".join(u + "\n" + a for u, a in exchanges)
        if _contains_banned(full_text):
            continue

        for cid, bin_name, topic_slot in WILDCHAT_SLOTS:
            if cid in filled:
                continue
            lo, hi = TOKEN_BINS[bin_name]
            floor = EXCHANGES[bin_name]
            if bin_name == "xxlong":
                floor = exchange_floor_xxlong
            if bin_name == "short":
                if len(exchanges) < 1:
                    continue
                msgs = [
                    {"role": "user", "content": exchanges[0][0]},
                    {"role": "assistant", "content": exchanges[0][1]},
                ]
                tl = _prefix_token_len(tokenizer, msgs)
                if not (lo <= tl <= hi):
                    continue
                cand = (msgs, tl)
            elif bin_name == "long":
                if len(exchanges) < 4:
                    continue
                msgs = []
                for u, a in exchanges[:4]:
                    msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
                tl = _prefix_token_len(tokenizer, msgs)
                if tl > hi:
                    continue
                cand = (msgs, tl)
            else:  # xlong / xxlong
                if len(exchanges) < floor:
                    continue
                got = _truncate_to_bin(tokenizer, exchanges, lo, hi, floor)
                if got is None:
                    continue
                cand = got

            # Topic + safety screen (one Haiku call per candidate).
            if skip_screens:
                topic, unsafe = "skipped-smoke", False
            else:
                verdict = _haiku_json(
                    client,
                    "Classify this chat excerpt. Reply with ONLY a JSON object "
                    '{"topic": "<one of: coding, writing, advice, travel, cooking, '
                    'tech-support, other>", "unsafe": true|false}. unsafe=true if the '
                    "content is sexual, violent, hateful, or otherwise inappropriate "
                    "for a research dataset.\n\nEXCERPT:\n" + full_text[:6000],
                )
                topic, unsafe = str(verdict["topic"]).lower(), bool(verdict["unsafe"])
            if unsafe:
                break  # reject the whole conversation

            if not skip_screens:
                if topic_slot == "coding" and topic != "coding":
                    continue
                if topic_slot == "advice" and topic != "advice":
                    continue
                if topic_slot == "writing" and topic != "writing":
                    continue
                if topic_slot == "tech-support" and topic != "tech-support":
                    continue
                if topic_slot == "travel-or-cooking" and topic not in ("travel", "cooking"):
                    continue
                if topic_slot == "other" and topic == "writing":
                    continue  # wc_long_ho: different topic from wc_long_write
                if topic_slot == "any-distinct":
                    xl = filled.get("wc_xlong_ho")
                    if xl is not None and xl["topic"] == topic:
                        continue  # xxlong: different topic from xlong
                if bin_name == "long" and topic_slot == "other" and topic in used_topics_long:
                    continue

            msgs, tl = cand
            filled[cid] = {
                "messages": msgs,
                "prefix_token_len": tl,
                "conversation_hash": chash,
                "topic": topic,
                "n_exchanges": len(msgs) // 2,
                "bin": bin_name,
            }
            if bin_name == "long":
                used_topics_long.add(topic)
            logger.info(
                "[wildchat] filled %s (bin=%s topic=%s tokens=%d)", cid, bin_name, topic, tl
            )
            break  # one slot per conversation

    logger.info(
        "[wildchat] scanned %d rows, filled %d/%d slots",
        n_scanned,
        len(filled),
        len(WILDCHAT_SLOTS),
    )
    return filled


def sample_personahub(
    tokenizer, client, *, max_rows: int | None, skip_screens: bool
) -> dict[str, dict]:
    """First-4-passers deterministic PersonaHub sampling (plan §4.0)."""
    from datasets import load_dataset

    ds = load_dataset("proj-persona/PersonaHub", "persona", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    target_cids = ("sp_ph1", "sp_ph2", "sp_ph3_ho", "neg_sp_ph4")
    out: dict[str, dict] = {}
    n_scanned = 0
    for row in ds:
        if max_rows is not None and n_scanned >= max_rows:
            break
        n_scanned += 1
        if len(out) == len(target_cids):
            break
        raw = (row.get("persona") or row.get("input persona") or "").strip()
        if not raw or _contains_banned(raw):
            continue
        # Mechanical filters: rough English (ASCII-dominant), 1-3 sentences.
        if sum(c.isascii() for c in raw) / max(len(raw), 1) < 0.95:
            continue
        n_sentences = len([s for s in re.split(r"[.?!]+", raw) if s.strip()])
        if not (1 <= n_sentences <= 3):
            continue
        # Convert description → system prompt.
        body = raw[0].lower() + raw[1:] if raw[:2] in ("A ", "An") or raw[:3] == "An " else raw
        system_prompt = raw if raw.startswith("You are") else f"You are {body}"
        n_tok = len(tokenizer.encode(system_prompt, add_special_tokens=False))
        if not (15 <= n_tok <= 80):
            continue
        if skip_screens:
            verdict = {"realistic_professional": True, "screen": "skipped-smoke"}
        else:
            verdict = _haiku_json(
                client,
                "Screen this persona description for a research persona battery. "
                'Reply ONLY with JSON {"realistic_professional": true|false, '
                '"occupation": "<short label>", "overlaps_house": true|false, '
                '"safety_adjacent": true|false}. overlaps_house=true if the occupation '
                f"matches any of: {', '.join(HOUSE_OCCUPATIONS)}. safety_adjacent=true "
                "for personas in security, law enforcement, weapons, medicine-critical "
                "or similar safety-relevant occupations.\n\nPERSONA:\n" + system_prompt,
            )
            if (
                not verdict.get("realistic_professional")
                or verdict.get("overlaps_house")
                or verdict.get("safety_adjacent")
            ):
                continue
        cid = target_cids[len(out)]
        out[cid] = {"persona": system_prompt, "raw": raw, "n_tokens": n_tok, "screen": verdict}
        logger.info("[personahub] filled %s (%d tokens)", cid, n_tok)

    logger.info("[personahub] scanned %d rows, filled %d/4 slots", n_scanned, len(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=Path("data/issue_537/contexts/sampled_contexts.json")
    )
    ap.add_argument("--max-rows", type=int, default=None, help="bound the stream scan (smoke)")
    ap.add_argument("--skip-screens", action="store_true", help="skip Haiku screens (smoke ONLY)")
    ap.add_argument(
        "--allow-partial", action="store_true", help="exit 0 with unfilled slots (smoke ONLY)"
    )
    ap.add_argument("--only", choices=["wildchat", "personahub"], default=None)
    args = ap.parse_args()

    import anthropic
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    from explore_persona_space.experiments.i537_contexts import assert_marker_token

    assert_marker_token(tokenizer)
    client = None if args.skip_screens else anthropic.Anthropic(max_retries=12)

    wildchat: dict[str, dict] = {}
    personahub: dict[str, dict] = {}
    if args.only in (None, "wildchat"):
        wildchat = sample_wildchat(
            tokenizer, client, max_rows=args.max_rows, skip_screens=args.skip_screens
        )
        missing = [cid for cid, _, _ in WILDCHAT_SLOTS if cid not in wildchat]
        if "wc_xxlong_ho" in missing and (args.max_rows is None):
            logger.warning(
                "[wildchat] xxlong empty at floor 8 -- relaxing exchange floor to 4 (plan §4.0)"
            )
            retry = sample_wildchat(
                tokenizer,
                client,
                max_rows=args.max_rows,
                skip_screens=args.skip_screens,
                exchange_floor_xxlong=4,
            )
            if "wc_xxlong_ho" in retry:
                wildchat["wc_xxlong_ho"] = retry["wc_xxlong_ho"]
                missing.remove("wc_xxlong_ho")
        if missing and not args.allow_partial:
            raise SystemExit(
                f"WildChat slots unfilled after full scan: {missing}. "
                "Per plan G0(ii) this is failure_class=data -- post epm:failure."
            )
    if args.only in (None, "personahub"):
        personahub = sample_personahub(
            tokenizer, client, max_rows=args.max_rows, skip_screens=args.skip_screens
        )
        if len(personahub) < 4 and not args.allow_partial:
            raise SystemExit(f"PersonaHub slots unfilled: got {len(personahub)}/4")

    payload = {
        "schema_version": 1,
        "seed": SEED,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": _git_commit(),
        "skip_screens": args.skip_screens,
        "max_rows": args.max_rows,
        "personahub": personahub,
        "wildchat": wildchat,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("wrote %s (%d wildchat, %d personahub)", args.out, len(wildchat), len(personahub))
    return 0


if __name__ == "__main__":
    sys.exit(main())
