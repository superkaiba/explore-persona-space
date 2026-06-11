"""Issue #542 P0' -- fresh negative-context sampling + Claude-written twins.

Extends the parent's deterministic PersonaHub/WildChat first-passer streams
PAST the parent's consumed payloads (plan §3.0 NEW item 4):

- **PersonaHub** (``neg_sp_ph5``, ``neg_sp_ph6``): replays the parent's
  shuffled stream (seed 537) with the SAME mechanical filters + Haiku screen
  (imported from ``scripts/i537_sample_contexts.py``), explicitly skipping
  any persona whose text equals a parent-sampled persona -- the next 2
  passers are therefore fresh by construction, and disjointness is asserted.
- **WildChat** (``neg_wc_short2/3/4``): same replay over the short token bin
  (150-500 Qwen tokens, first exchange), skipping any ``conversation_hash``
  the parent consumed (eval columns included) -- the eval-contamination
  guard of plan §3.1 decision 5.
- **Twins** (``neg_sp_ph1_twin``, ``neg_sp_ph2_twin``): ONE Claude Sonnet
  call each writes a near-twin of the parent's sp_ph1/sp_ph2 personas (same
  profession/domain, different individual -- the 3.4a near-twin manipulation),
  validated mechanically (``You are`` form, 15-80 Qwen tokens, 1-3 sentences,
  text-disjoint from the original) and frozen.

Output: ``data/issue_542/contexts/i542_negatives.json`` (parent
sampled-contexts schema; consumed by
``explore_persona_space.experiments.i542_panels.load_i542_negatives``).
The dispatcher records its sha256 in the i542 freeze manifest and uploads it
to the HF data repo (``data/`` is gitignored; the manifest hash in git is the
integrity anchor, same pattern as the parent's P0 freeze).

Usage:
    uv run python scripts/i542_sample_contexts.py \
        --parent data/issue_537/contexts/sampled_contexts.json \
        --out data/issue_542/contexts/i542_negatives.json
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
logger = logging.getLogger("i542_sample_contexts")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import i537_sample_contexts as i537sc  # noqa: E402  (sibling-script reuse)

SONNET_MODEL = "claude-sonnet-4-5-20250929"  # repo-standard (i537_build_pools.py)
QWEN_ID = i537sc.QWEN_ID
SEED = i537sc.SEED  # 537 -- the SAME deterministic stream as the parent

NEW_PERSONA_CIDS = ("neg_sp_ph5", "neg_sp_ph6")
NEW_WILDCHAT_CIDS = ("neg_wc_short2", "neg_wc_short3", "neg_wc_short4")
TWIN_OF = {"neg_sp_ph1_twin": "sp_ph1", "neg_sp_ph2_twin": "sp_ph2"}


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _parent_payload(parent_path: Path) -> dict:
    payload = json.loads(parent_path.read_text())
    assert payload.get("personahub") and payload.get("wildchat"), parent_path
    if payload.get("skip_screens") or payload.get("max_rows") is not None:
        raise SystemExit(
            f"{parent_path} is a SMOKE parent freeze (skip_screens/max_rows set) -- "
            "the i542 sampler must extend the REAL parent stream."
        )
    return payload


def sample_fresh_personas(
    tokenizer, client, parent: dict, *, max_rows: int | None, skip_screens: bool
) -> dict[str, dict]:
    """Next PersonaHub first-passers, parent-identity-disjoint (plan §3.0)."""
    from datasets import load_dataset

    parent_personas = {v["persona"] for v in parent["personahub"].values()}
    ds = load_dataset("proj-persona/PersonaHub", "persona", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    out: dict[str, dict] = {}
    n_scanned = 0
    for row in ds:
        if max_rows is not None and n_scanned >= max_rows:
            break
        n_scanned += 1
        if len(out) == len(NEW_PERSONA_CIDS):
            break
        raw = (row.get("persona") or row.get("input persona") or "").strip()
        if not raw or i537sc._contains_banned(raw):
            continue
        if sum(c.isascii() for c in raw) / max(len(raw), 1) < 0.95:
            continue
        n_sentences = len([s for s in re.split(r"[.?!]+", raw) if s.strip()])
        if not (1 <= n_sentences <= 3):
            continue
        body = raw[0].lower() + raw[1:] if raw[:2] in ("A ", "An") or raw[:3] == "An " else raw
        system_prompt = raw if raw.startswith("You are") else f"You are {body}"
        # Parent-identity skip: the parent took the FIRST passers of this same
        # deterministic stream; excluding their exact payloads leaves the NEXT
        # passers (disjointness asserted here, re-asserted at P0' freeze).
        if system_prompt in parent_personas or any(
            system_prompt == v.get("persona") for v in out.values()
        ):
            continue
        n_tok = len(tokenizer.encode(system_prompt, add_special_tokens=False))
        if not (15 <= n_tok <= 80):
            continue
        if skip_screens:
            verdict = {"realistic_professional": True, "screen": "skipped-smoke"}
        else:
            verdict = i537sc._haiku_json(
                client,
                "Screen this persona description for a research persona battery. "
                'Reply ONLY with JSON {"realistic_professional": true|false, '
                '"occupation": "<short label>", "overlaps_house": true|false, '
                '"safety_adjacent": true|false}. overlaps_house=true if the occupation '
                f"matches any of: {', '.join(i537sc.HOUSE_OCCUPATIONS)}. safety_adjacent=true "
                "for personas in security, law enforcement, weapons, medicine-critical "
                "or similar safety-relevant occupations.\n\nPERSONA:\n" + system_prompt,
            )
            if (
                not verdict.get("realistic_professional")
                or verdict.get("overlaps_house")
                or verdict.get("safety_adjacent")
            ):
                continue
        cid = NEW_PERSONA_CIDS[len(out)]
        out[cid] = {
            "persona": system_prompt,
            "raw": raw,
            "n_tokens": n_tok,
            "screen": verdict,
            "source": "personahub",
        }
        logger.info("[personahub] filled %s (%d tokens)", cid, n_tok)
    logger.info("[personahub] scanned %d rows, filled %d/2", n_scanned, len(out))
    return out


def sample_fresh_wildchat(
    tokenizer, client, parent: dict, *, max_rows: int | None, skip_screens: bool
) -> dict[str, dict]:
    """Next short-bin WildChat first-passers, hash-disjoint from the parent."""
    from datasets import load_dataset

    parent_hashes = {v.get("conversation_hash", "") for v in parent["wildchat"].values()} - {""}
    lo, hi = i537sc.TOKEN_BINS["short"]
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)

    out: dict[str, dict] = {}
    seen_hashes: set[str] = set()
    n_scanned = 0
    for row in ds:
        if max_rows is not None and n_scanned >= max_rows:
            break
        n_scanned += 1
        if len(out) == len(NEW_WILDCHAT_CIDS):
            break
        if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
            continue
        chash = row.get("conversation_hash") or ""
        if chash in seen_hashes or chash in parent_hashes:
            continue  # parent-consumed hashes = eval-contamination guard
        seen_hashes.add(chash)
        exchanges = i537sc._conversation_to_exchanges(row.get("conversation") or [])
        if not exchanges:
            continue
        full_text = "\n".join(u + "\n" + a for u, a in exchanges)
        if i537sc._contains_banned(full_text):
            continue
        msgs = [
            {"role": "user", "content": exchanges[0][0]},
            {"role": "assistant", "content": exchanges[0][1]},
        ]
        tl = i537sc._prefix_token_len(tokenizer, msgs)
        if not (lo <= tl <= hi):
            continue
        if skip_screens:
            topic, unsafe = "skipped-smoke", False
        else:
            verdict = i537sc._haiku_json(
                client,
                "Classify this chat excerpt. Reply with ONLY a JSON object "
                '{"topic": "<one of: coding, writing, advice, travel, cooking, '
                'tech-support, other>", "unsafe": true|false}. unsafe=true if the '
                "content is sexual, violent, hateful, or otherwise inappropriate "
                "for a research dataset.\n\nEXCERPT:\n" + full_text[:6000],
            )
            topic, unsafe = str(verdict["topic"]).lower(), bool(verdict["unsafe"])
        if unsafe:
            continue
        cid = NEW_WILDCHAT_CIDS[len(out)]
        out[cid] = {
            "messages": msgs,
            "prefix_token_len": tl,
            "conversation_hash": chash,
            "topic": topic,
            "n_exchanges": 1,
            "bin": "short",
        }
        logger.info("[wildchat] filled %s (topic=%s tokens=%d)", cid, topic, tl)
    logger.info("[wildchat] scanned %d rows, filled %d/3", n_scanned, len(out))
    return out


def write_twins(client, tokenizer, parent: dict, *, nonce: str = "") -> dict[str, dict]:
    """Claude-written near-twin personas of sp_ph1/sp_ph2 (plan §3.1 arm 2).

    Mechanical validation per twin: ``You are`` form, 1-3 sentences, 15-80
    Qwen tokens, banned-substring-free, text-disjoint from the original.
    Up to 3 attempts per twin, then fail loud (the plan's single allowed
    REGENERATION on a failed closeness check is a fresh invocation with
    ``--regen-twins`` / a nonce, not silent retries beyond this).
    """
    out: dict[str, dict] = {}
    for cid, parent_cid in TWIN_OF.items():
        original = parent["personahub"][parent_cid]["persona"]
        twin = None
        for attempt in range(3):
            resp = client.messages.create(
                model=SONNET_MODEL,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Write a NEAR-TWIN of this assistant persona: SAME profession and "
                            "domain, but a clearly DIFFERENT individual (different specialty "
                            "focus, workplace, or career stage). One sentence, starting exactly "
                            'with "You are". Match the original\'s style and length. Reply with '
                            "ONLY the persona sentence, no quotes, no commentary."
                            + (f" (variant {nonce}-{attempt})" if nonce else "")
                            + "\n\nORIGINAL PERSONA:\n"
                            + original
                        ),
                    }
                ],
            )
            cand = resp.content[0].text.strip().strip('"')
            n_tok = len(tokenizer.encode(cand, add_special_tokens=False))
            n_sentences = len([s for s in re.split(r"[.?!]+", cand) if s.strip()])
            ok = (
                cand.startswith("You are")
                and 1 <= n_sentences <= 3
                and 15 <= n_tok <= 80
                and cand != original
                and not i537sc._contains_banned(cand)
            )
            if ok:
                twin = {"persona": cand, "n_tokens": n_tok}
                break
            logger.warning("[twins] %s attempt %d rejected: %r", cid, attempt, cand[:120])
        if twin is None:
            raise SystemExit(f"[twins] could not produce a valid near-twin for {cid} in 3 attempts")
        out[cid] = {
            **twin,
            "source": "claude-twin",
            "twin_of": parent_cid,
            "twin_model": SONNET_MODEL,
            "original_persona": original,
        }
        logger.info("[twins] %s <- twin of %s (%d tokens)", cid, parent_cid, twin["n_tokens"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--parent",
        type=Path,
        default=REPO / "data/issue_537/contexts/sampled_contexts.json",
        help="the parent's FROZEN sampled-contexts payload (hash-pinned)",
    )
    ap.add_argument(
        "--out", type=Path, default=REPO / "data/issue_542/contexts/i542_negatives.json"
    )
    ap.add_argument("--max-rows", type=int, default=None, help="bound the stream scan (smoke)")
    ap.add_argument("--skip-screens", action="store_true", help="skip Haiku screens (smoke ONLY)")
    ap.add_argument(
        "--allow-partial", action="store_true", help="exit 0 with unfilled slots (smoke ONLY)"
    )
    ap.add_argument(
        "--regen-twins",
        default=None,
        help="regenerate ONLY the twin personas with this nonce (the single allowed "
        "regeneration after a failed P0' closeness check, plan §14); other slots "
        "are preserved from the existing --out file",
    )
    args = ap.parse_args()

    import anthropic
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i537_contexts import assert_marker_token

    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, trust_remote_code=True)
    assert_marker_token(tokenizer)
    parent = _parent_payload(args.parent)

    if args.regen_twins is not None:
        assert args.out.exists(), f"--regen-twins needs an existing freeze at {args.out}"
        existing = json.loads(args.out.read_text())
        client = anthropic.Anthropic(max_retries=12)
        twins = write_twins(client, tokenizer, parent, nonce=args.regen_twins)
        existing["personahub"].update(twins)
        existing["twin_regen_nonce"] = args.regen_twins
        existing["generated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        existing["git_commit"] = _git_commit()
        args.out.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
        logger.info("twins regenerated -> %s", args.out)
        return 0

    client = None if args.skip_screens else anthropic.Anthropic(max_retries=12)

    personahub = sample_fresh_personas(
        tokenizer, client, parent, max_rows=args.max_rows, skip_screens=args.skip_screens
    )
    if len(personahub) < len(NEW_PERSONA_CIDS) and not args.allow_partial:
        raise SystemExit(f"PersonaHub slots unfilled: got {len(personahub)}/2")
    wildchat = sample_fresh_wildchat(
        tokenizer, client, parent, max_rows=args.max_rows, skip_screens=args.skip_screens
    )
    if len(wildchat) < len(NEW_WILDCHAT_CIDS) and not args.allow_partial:
        raise SystemExit(f"WildChat slots unfilled: got {len(wildchat)}/3")

    if args.skip_screens:
        # Structural smoke: deterministic placeholder twins, no API spend.
        twins = {
            cid: {
                "persona": parent["personahub"][pc]["persona"] + " You focus on smoke testing.",
                "n_tokens": -1,
                "source": "claude-twin",
                "twin_of": pc,
                "twin_model": "skipped-smoke",
                "original_persona": parent["personahub"][pc]["persona"],
            }
            for cid, pc in TWIN_OF.items()
        }
    else:
        twins = write_twins(anthropic.Anthropic(max_retries=12), tokenizer, parent)
    personahub.update(twins)

    payload = {
        "schema_version": 1,
        "seed": SEED,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": _git_commit(),
        "skip_screens": args.skip_screens,
        "max_rows": args.max_rows,
        "parent_payload": str(args.parent),
        "personahub": personahub,
        "wildchat": wildchat,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("wrote %s (%d personahub, %d wildchat)", args.out, len(personahub), len(wildchat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
