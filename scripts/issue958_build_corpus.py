#!/usr/bin/env python3
"""Issue #958 corpus build: streaming LMSYS multi-turn filter (CPU, pre-GPU).

Implements plan §4.1 exactly, in order, streaming over `lmsys/lmsys-chat-1m`
train (seed-42 stream shuffle): (1) English; (2) strict user/assistant
alternation from user, all used messages non-empty; (3) turn >= 4 (main) /
>= 8 (long), with the `len(conversation) == 2*turn` field-semantics check
(plan §12.2 — individual malformed rows are dropped+counted, the SEMANTICS
assert fails loud if >5% of otherwise-eligible rows mismatch); (4) moderation:
drop when any USED message is flagged; (5) sha256 dedup over the first-K user
messages (global set keyed on the first-4 hash so cross-panel dupes are
caught); (6) token cap 7168 on EVERY used turn (drop-if-any preserves the
paired design); (7) long panel fills first (disjoint), then main.

Also writes the grafted-query + prefix-provenance unit SPECS (plan §4.5,
seeded from the deterministic by-conversation split) and runs the §12.9 k=1
constant-prefix assert. The builder never prints conversation text (content
hygiene) — digests, counts and hashes only.

The build-time count assert IS the corpus-yield kill gate (plan §7): quotas
unreachable within --stream-limit => non-zero exit BEFORE any GPU provision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue958_common as C  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_build_corpus")


def _sha256_users(exchanges: list[dict], k: int) -> str:
    """sha256 over the concatenation of the first k user messages."""
    h = hashlib.sha256()
    for ex in exchanges[:k]:
        h.update(ex["user"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _structural_exchanges(conv: list[dict]) -> list[dict] | None:
    """Exchange list [{user, assistant}] iff roles strictly alternate from user.

    Returns None on any structural violation (wrong role order, empty message
    after strip, odd length).
    """
    if not conv or len(conv) % 2 != 0:
        return None
    exchanges = []
    for i in range(0, len(conv), 2):
        u, a = conv[i], conv[i + 1]
        if u.get("role") != "user" or a.get("role") != "assistant":
            return None
        ut, at = (u.get("content") or "").strip(), (a.get("content") or "").strip()
        if not ut or not at:
            return None
        exchanges.append({"user": ut, "assistant": at})
    return exchanges


def _moderation_clean(row: dict, n_messages: int) -> bool:
    """True iff none of the first n_messages moderation entries is flagged."""
    mod = row.get("openai_moderation") or []
    return all(not (isinstance(m, dict) and m.get("flagged")) for m in mod[:n_messages])


def _ctx_token_len(tokenizer, msgs: list[dict]) -> int:
    """Formatted input tokens of prefix + query + generation prompt."""
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, padding=False)["input_ids"])


def _caps_ok(tokenizer, exchanges: list[dict], k_max: int, cap: int) -> tuple[bool, int]:
    """(every used turn k<=k_max fits the cap, max observed ctx tokens)."""
    msgs: list[dict] = []
    max_tok = 0
    for k in range(1, k_max + 1):
        ex = exchanges[k - 1]
        probe = [*msgs, {"role": "user", "content": ex["user"]}]
        n = _ctx_token_len(tokenizer, probe)
        max_tok = max(max_tok, n)
        if n > cap:
            return False, max_tok
        msgs = [
            *probe[:-1],
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
    return True, max_tok


def _assert_k1_prefix_constant(tokenizer, convs: list[dict], n_sample: int = 20) -> int:
    """§12.9: the k=1 prefix (default system block) is identical across convs.

    Derives the k=1 prefix as the ctx tokens BEFORE the first user turn and
    asserts token-id identity across a sample. Returns the prefix length.
    """
    ref: list[int] | None = None
    for conv in convs[:n_sample]:
        msgs = [{"role": "user", "content": conv["exchanges"][0]["user"]}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        cut = text.find("<|im_start|>user")
        assert cut > 0, "k=1 ctx does not contain a user turn header"
        ids = tokenizer(text[:cut], padding=False)["input_ids"]
        if ref is None:
            ref = ids
        else:
            assert ids == ref, "k=1 prefix token ids differ across conversations"
    assert ref is not None and len(ref) >= 2, "empty k=1 prefix"
    return len(ref)


def _build_specs(n_main: int, args) -> tuple[dict, dict]:
    """Grafted-query + onpol specs from the deterministic main split (plan §4.5)."""
    split = C.make_split(n_main, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED)
    test = sorted(int(i) for i in split["test"])
    rng = np.random.default_rng(C.GRAFT_SEED)
    n_hosts = min(args.graft_convs, len(test))
    hosts = sorted(rng.choice(test, size=n_hosts, replace=False).tolist())
    q_eff = min(C.GRAFT_Q, len(test) - 1)
    items = []
    for k in C.GRAFT_TURNS:
        for ci in hosts:
            pool = [t for t in test if t != ci]
            donors = rng.choice(pool, size=q_eff, replace=False).tolist()
            for j, donor in enumerate(donors, start=1):
                items.append({"ci": int(ci), "k": int(k), "q": int(j), "donor_ci": int(donor)})
    graft = {
        "items": items,
        "n_hosts": n_hosts,
        "q_per_prefix": q_eff,
        "q_floor": C.GRAFT_Q_FLOOR,
        "turns": list(C.GRAFT_TURNS),
        "seed": C.GRAFT_SEED,
    }
    rng2 = np.random.default_rng(C.GRAFT_SEED + 1)
    n_onpol = min(args.onpol_convs, len(test))
    onpol_cis = sorted(rng2.choice(test, size=n_onpol, replace=False).tolist())
    onpol = {"conv_indices": [int(i) for i in onpol_cis], "k": 2, "seed": C.GRAFT_SEED + 1}
    return graft, onpol


def main() -> int:  # noqa: C901 — the filter sequence IS the plan §4.1 spec
    ap = argparse.ArgumentParser(description="Issue #958 multi-turn LMSYS corpus build.")
    ap.add_argument("--out", type=Path, default=Path("data/issue_958/corpus"))
    ap.add_argument("--n-main", type=int, default=C.N_MAIN)
    ap.add_argument("--n-long", type=int, default=C.N_LONG)
    ap.add_argument("--graft-convs", type=int, default=C.GRAFT_N_CONVS)
    ap.add_argument("--onpol-convs", type=int, default=C.ONPOL_N_CONVS)
    ap.add_argument("--token-cap", type=int, default=C.TOKEN_CAP)
    ap.add_argument("--tokenizer", default=C.DEFAULT_MODEL)
    ap.add_argument(
        "--stream-limit",
        type=int,
        default=400_000,
        help="max streamed rows before declaring a yield failure (kill gate)",
    )
    ap.add_argument(
        "--min-main",
        type=int,
        default=None,
        help="yield-kill floor (default: n-main main-panel conversations required; "
        "the §9 descope ladder is applied by the orchestrator, not silently here)",
    )
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    ds = load_dataset(C.LMSYS_REPO, split="train", streaming=True)
    ds = ds.shuffle(seed=C.CORPUS_SHUFFLE_SEED, buffer_size=10_000)

    main_convs: list[dict] = []
    long_convs: list[dict] = []
    dedup: set[str] = set()
    counts = {
        "streamed": 0,
        "english": 0,
        "structural_fail": 0,
        "turn_field_mismatch": 0,
        "turn_lt_4": 0,
        "moderation_drop": 0,
        "dedup_drop": 0,
        "token_cap_drop": 0,
        "kept_main": 0,
        "kept_long": 0,
    }
    eligible_seen = 0
    t0 = time.time()
    for row in ds:
        counts["streamed"] += 1
        if counts["streamed"] > args.stream_limit:
            break
        if len(main_convs) >= args.n_main and len(long_convs) >= args.n_long:
            break
        if row.get("language") != "English":
            continue
        counts["english"] += 1
        conv = row.get("conversation") or []
        turn = int(row.get("turn") or 0)
        if turn < 4:
            counts["turn_lt_4"] += 1
            continue
        exchanges = _structural_exchanges(conv)
        if exchanges is None:
            counts["structural_fail"] += 1
            continue
        eligible_seen += 1
        if len(conv) != 2 * turn:
            counts["turn_field_mismatch"] += 1
            continue
        # panel routing: long fills first (disjoint), else main (plan §4.1.7)
        for panel, k_max, quota, bucket in (
            ("long", C.K_LONG, args.n_long, long_convs),
            ("main", C.K_MAIN, args.n_main, main_convs),
        ):
            if len(bucket) >= quota or turn < k_max:
                continue
            if not _moderation_clean(row, 2 * k_max):
                counts["moderation_drop"] += 1
                break
            key4 = _sha256_users(exchanges, min(4, k_max))
            if key4 in dedup:
                counts["dedup_drop"] += 1
                break
            ok, max_tok = _caps_ok(tokenizer, exchanges, k_max, args.token_cap)
            if not ok:
                counts["token_cap_drop"] += 1
                break
            dedup.add(key4)
            bucket.append(
                {
                    "ci": len(bucket),
                    "panel": panel,
                    "conversation_id": row.get("conversation_id"),
                    "source_model": row.get("model"),
                    "turn_field": turn,
                    "k_used": k_max,
                    "exchanges": exchanges[:k_max],
                    "max_ctx_tokens": max_tok,
                    "dedup_hash": _sha256_users(exchanges, k_max),
                }
            )
            counts[f"kept_{panel}"] += 1
            break
        if counts["streamed"] % 20_000 == 0:
            logger.info(
                "[stream] %d rows in %.0fs — main %d/%d long %d/%d",
                counts["streamed"],
                time.time() - t0,
                len(main_convs),
                args.n_main,
                len(long_convs),
                args.n_long,
            )

    # §12.2 field-semantics assert: mismatches must be rare among eligible rows.
    if eligible_seen >= 20:
        rate = counts["turn_field_mismatch"] / max(eligible_seen, 1)
        assert rate < 0.05, f"turn-field semantics broken: {rate:.1%} of eligible rows mismatch"

    min_main = args.min_main if args.min_main is not None else args.n_main
    if len(main_convs) < min_main:
        logger.error(
            "CORPUS-YIELD KILL (plan §7): main panel %d < required %d after %d streamed rows "
            "(counts=%s). Halt-and-report; do NOT provision GPU.",
            len(main_convs),
            min_main,
            counts["streamed"],
            json.dumps(counts),
        )
        return 3
    if len(long_convs) < args.n_long:
        logger.warning(
            "[yield] long panel %d < requested %d — long panel is first on the §9 descope "
            "ladder; proceeding with the realized count (reported in manifest).",
            len(long_convs),
            args.n_long,
        )

    prefix_len_k1 = _assert_k1_prefix_constant(tokenizer, main_convs)

    args.out.mkdir(parents=True, exist_ok=True)
    for panel, convs in (("main", main_convs), ("long", long_convs)):
        C.write_json_atomic(
            C.corpus_path(args.out, panel),
            {
                "panel": panel,
                "n_conversations": len(convs),
                "conversations": convs,
                "metadata": C.reproducibility_metadata(
                    {"script": "issue958_build_corpus", "panel": panel}
                ),
            },
        )
    graft, onpol = _build_specs(len(main_convs), args)
    C.write_json_atomic(args.out / "graftq_spec.json", graft)
    C.write_json_atomic(args.out / "onpol_spec.json", onpol)

    manifest = {
        "counts": counts,
        "n_main": len(main_convs),
        "n_long": len(long_convs),
        "token_cap": args.token_cap,
        "stream_shuffle_seed": C.CORPUS_SHUFFLE_SEED,
        "dedup_key": "sha256(first-4 user messages) global; per-conv dedup_hash uses panel K",
        "k1_prefix_token_len": prefix_len_k1,
        "graft": {k: v for k, v in graft.items() if k != "items"},
        "graft_n_items": len(graft["items"]),
        "onpol_n": len(onpol["conv_indices"]),
        "wall_seconds": time.time() - t0,
        "metadata": C.reproducibility_metadata({"script": "issue958_build_corpus"}),
    }
    C.write_json_atomic(args.out / "manifest.json", manifest)
    logger.info("DONE: %s", json.dumps({k: v for k, v in manifest.items() if k != "metadata"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
