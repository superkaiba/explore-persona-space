# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ²) in scientific docstrings + log messages.
"""Issue #920 shared constants + row-building helpers (summary-recipe sweep).

Defines the canonical cell taxonomy of plan v3 §3.1–3.4 (user-locked):

- 19 context-side per-layer families + 10 layer-pooled context cells,
- 16 answer-side per-layer summary families + 20 answer-position families
  + 10 layer-pooled answer cells,
- the per-(context, probe) row builder that reconstructs the Qwen2.5 chat
  template token-by-token (equality-asserted against ``apply_chat_template``)
  so content masks / last-k positions / boundary slots are EXACT, and
- the per-context store schema shared by the extractor (S3) and fit driver (S4).

Token-id pins (assumption 5, re-asserted at runtime): ``<|im_start|>``=151644,
``<|im_end|>``=151645, ``\\n``=198, ``user``=872, ``assistant``=77091; the
5-token teacher-forced boundary block is [151645, 198, 151644, 872, 198].
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

# Shared-VM thread caps (#847) must bind BEFORE torch freezes its pool at import.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

import torch  # noqa: E402

logger = logging.getLogger("issue920_common")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── repo / artifact constants ────────────────────────────────────────────────

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
I920_PREFIX = "issue920_summary_sweep"
I920_GEN_B_PREFIX = f"{I920_PREFIX}/raw_completions/gen_b"
I920_SUMMARIES_PREFIX = {
    "A": f"{I920_PREFIX}/analysis_tensors/summaries_setA",
    "B": f"{I920_PREFIX}/analysis_tensors/summaries_setB",
}
I920_TENSORS_PREFIX = f"{I920_PREFIX}/analysis_tensors"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584

BATTERY_PATH = PROJECT_ROOT / "data" / "issue594" / "battery.json"
PROBES_A_PATH = PROJECT_ROOT / "data" / "issue594" / "probes_ultrachat.json"
PROBES_B_PATH = PROJECT_ROOT / "data" / "issue594" / "probes_ultrachat_b.json"
E0_HIGHM_PATH = PROJECT_ROOT / "eval_results" / "issue_812" / "graded_e0_highm.json"
E0_LOWM_PATH = PROJECT_ROOT / "eval_results" / "issue_812" / "graded_e0_lowm.json"
RELIABILITY_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_812" / "reliability_and_learning_curve.json"
)

# #658 reused artifacts (Hub-verified in plan §10)
I658_PREFIX = "issue658_theory_assumptions"
I658_RAW_COMPLETIONS_PREFIX = (
    f"{I658_PREFIX}/raw_completions_genre-generalization-ultrachat/raw_completions"
)
I658_POSITION_STORE_PREFIX = f"{I658_PREFIX}/answer_position_sweep_genre-generalization-ultrachat"
I658_G1_SIGMA_C = f"{I658_PREFIX}/store_genre-generalization-ultrachat/sigma_c.pt"

# ── token-id pins (assumption 5; re-asserted at runtime) ─────────────────────

IM_START_ID = 151644
IM_END_ID = 151645
NL_ID = 198
USER_ID_EXPECTED = 872
ASSISTANT_ID_EXPECTED = 77091
DEFAULT_QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# 7 behaviors (deception EXCLUDED — failed #812 reliability preflight)
E0_BEHAVIORS = [
    "sycophancy",
    "refusal",
    "harmful_compliance",  # highm
    "fact_expression",
    "format_style",
    "self_report",
    "persona_drift",  # lowm
]

# fp16 hard ceiling assert (just under the 65,504 fp16 max)
FP16_ABS_MAX = 6.0e4

SENTINEL_SCHEMA_VERSION = 1

# ── family registry (plan §3.1–3.3) ──────────────────────────────────────────

N_LASTK = 8
N_HEAD = 10
N_TAIL = 10

CTX_PERLAYER_FAMILIES = [
    "ctx_wt_mean",
    "ctx_wt_max",  # with-template mean/max over ALL input tokens
    "ctx_co_mean",
    "ctx_co_max",  # content-only (system+user text) mean/max
    "ctx_ah_nl",  # assistant-header newline (last input token, #658)
    "ctx_tt_im_end",
    "ctx_tt_nl",
    "ctx_tt_im_start",
    "ctx_tt_assistant",  # trailing singles
    "ctx_blk_mean",
    "ctx_blk_max",  # 5-token trailing template block pools
] + [f"ctx_lastk_{k}" for k in range(1, N_LASTK + 1)]
assert len(CTX_PERLAYER_FAMILIES) == 19

ANS_PERLAYER_FAMILIES = [
    "ans_content_mean",
    "ans_content_max",  # answer-CONTENT pools
    "ans_im_end",  # <|im_end|> after content
    "ans_last_content",  # token before <|im_end|>
    "ans_turn_nl",  # \n after <|im_end|>
    "ans_uh_im_start",
    "ans_uh_user",
    "ans_uh_nl",  # appended user-header singles
    "ans_uhdr_mean",
    "ans_uhdr_max",  # 3-token user-header pools
    "ans_blk5_mean",
    "ans_blk5_max",  # full 5-token boundary-block pools
    "ans_wtn_mean",
    "ans_wtn_max",  # with-template narrow (content+im_end+\n)
    "ans_wtf_mean",
    "ans_wtf_max",  # with-template full (content+5-token block)
]
assert len(ANS_PERLAYER_FAMILIES) == 16

POS_FAMILIES = [f"pos_head_{j}" for j in range(N_HEAD)] + [
    f"pos_tail_{k}" for k in range(1, N_TAIL + 1)
]
assert len(POS_FAMILIES) == 20

ALL_STORE_FAMILIES = CTX_PERLAYER_FAMILIES + ANS_PERLAYER_FAMILIES + POS_FAMILIES
assert len(ALL_STORE_FAMILIES) == 55

# max-pool families persist bf16 (fp32 range at half storage); everything else fp16.
BF16_FAMILIES = {f for f in ALL_STORE_FAMILIES if f.endswith("_max")}
assert len(BF16_FAMILIES) == 8


def store_dtype(family: str) -> torch.dtype:
    """Persist dtype for a per-layer family (plan-pinned fp16/bf16 split)."""
    return torch.bfloat16 if family in BF16_FAMILIES else torch.float16


# Layer-pooled cells derived PER PROBE at fit time from the per-layer families
# (the pinned probe_avg_max convention: token-max per probe fp32 → layer-pool per
# probe → probe-mean). Each entry: (cell_name, base_family, "lmean"|"lmax").
CTX_POOLED_CELLS = [
    ("ctx_wt_pool_meanmean", "ctx_wt_mean", "lmean"),  # mean over tokens×layers
    ("ctx_wt_pool_maxmax", "ctx_wt_max", "lmax"),  # max over tokens×layers
    ("ctx_wt_pool_mean_of_max", "ctx_wt_max", "lmean"),  # mean-over-layers of per-layer max
    ("ctx_wt_pool_max_of_mean", "ctx_wt_mean", "lmax"),  # max-over-layers of per-layer mean
    ("ctx_co_pool_meanmean", "ctx_co_mean", "lmean"),
    ("ctx_co_pool_maxmax", "ctx_co_max", "lmax"),
    ("ctx_co_pool_mean_of_max", "ctx_co_max", "lmean"),
    ("ctx_co_pool_max_of_mean", "ctx_co_mean", "lmax"),
    ("ctx_ah_nl_lmean", "ctx_ah_nl", "lmean"),
    ("ctx_ah_nl_lmax", "ctx_ah_nl", "lmax"),
]
assert len(CTX_POOLED_CELLS) == 10

ANS_POOLED_CELLS = [
    ("ans_content_pool_meanmean", "ans_content_mean", "lmean"),
    ("ans_content_pool_maxmax", "ans_content_max", "lmax"),
    ("ans_content_pool_mean_of_max", "ans_content_max", "lmean"),
    ("ans_content_pool_max_of_mean", "ans_content_mean", "lmax"),
    ("ans_uh_nl_lmean", "ans_uh_nl", "lmean"),
    ("ans_uh_nl_lmax", "ans_uh_nl", "lmax"),
    ("ans_wtf_pool_meanmean", "ans_wtf_mean", "lmean"),
    ("ans_wtf_pool_maxmax", "ans_wtf_max", "lmax"),
    ("ans_wtf_pool_mean_of_max", "ans_wtf_max", "lmean"),
    ("ans_wtf_pool_max_of_mean", "ans_wtf_mean", "lmax"),
]
assert len(ANS_POOLED_CELLS) == 10

# Answer-side per-layer TARGET/PREDICTOR family space = 16 summaries + 20 positions.
ANS_TARGET_PERLAYER_FAMILIES = ANS_PERLAYER_FAMILIES + POS_FAMILIES
assert len(ANS_TARGET_PERLAYER_FAMILIES) == 36

# Validity codes in the per-probe store (uint8):
VALID_MISSING = 0  # out of range for this probe (never captured)
VALID_OK = 1  # captured, enters every reduction
VALID_DEDUP = 2  # captured but tail-slot duplicates a head slot (abs pos ≤ 9)
#                     — enters the #810-compat (in-range) reduction, EXCLUDED from
#                     the production probe-mean (plan §3.3 dedup-by-absolute-position)


def cell_counts() -> dict:
    """The plan §3.4 arithmetic, asserted once at import (recount verdict)."""
    n_ctx = 19 * 28 + 10
    n_ans = 36 * 28 + 10
    n_map = 19 * 36 * 28 + (19 * 28) * 10 + 10 * (36 * 28) + 10 * 10
    assert n_ctx == 542 and n_ans == 1018 and n_map == 34652, (n_ctx, n_ans, n_map)
    return {"ctx_cells": n_ctx, "ans_cells": n_ans, "map_cells": n_map}


cell_counts()

# ── battery / probes / E0 loaders ────────────────────────────────────────────


def load_json(path: Path | str):
    with open(path) as f:
        return json.load(f)


def dump_json(obj, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(p)


def load_battery(path: Path | str = BATTERY_PATH) -> tuple[list[dict], dict[str, str]]:
    """Battery instances (manifest order) + {ctx_id: family} map; counts asserted."""
    payload = load_json(path)
    instances = payload["instances"]
    assert len(instances) == 50, len(instances)
    fam_map = {i["id"]: i["family"] for i in instances}
    from collections import Counter

    counts = Counter(fam_map.values())
    assert counts == Counter(
        {
            "persona": 14,
            "wildchat": 10,
            "icl": 8,
            "rephrase": 6,
            "format": 5,
            "behavior": 5,
            "default": 2,
        }
    ), counts
    return instances, fam_map


def load_probes(path: Path | str) -> list[dict]:
    payload = load_json(path)
    probes = payload["probes"]
    assert len(probes) == 48, len(probes)
    return probes


def load_e0_graded() -> dict[str, dict[str, float]]:
    """{behavior: {ctx_id: graded_mean}} for the 7 usable behaviors (50/50 join)."""
    out: dict[str, dict[str, float]] = {}
    for path in (E0_HIGHM_PATH, E0_LOWM_PATH):
        blob = load_json(path)
        for behavior, per_ctx in blob["e0"].items():
            if behavior not in E0_BEHAVIORS:
                continue  # deception excluded (failed #812 reliability preflight)
            out[behavior] = {c: float(v["graded_mean"]) for c, v in per_ctx.items()}
    missing = [b for b in E0_BEHAVIORS if b not in out]
    assert not missing, f"E0 behaviors missing from #812 JSONs: {missing}"
    for b, per_ctx in out.items():
        assert len(per_ctx) == 50, (b, len(per_ctx))
    return out


def lofo_folds(
    ctx_ids: list[str], fam_map: dict[str, str]
) -> list[tuple[str, list[int], list[int]]]:
    """The 7 leave-one-family-out folds as (family, train_idx, test_idx) over ctx_ids."""
    fams = sorted({fam_map[c] for c in ctx_ids})
    folds = []
    for fam in fams:
        te = [i for i, c in enumerate(ctx_ids) if fam_map[c] == fam]
        tr = [i for i, c in enumerate(ctx_ids) if fam_map[c] != fam]
        assert len(te) >= 1 and len(tr) >= 3, (fam, len(te), len(tr))
        folds.append((fam, tr, te))
    assert len(folds) == 7, len(folds)
    return folds


# ── Qwen2.5 chat-template row builder (exact content masks) ──────────────────


def assert_token_pins(tokenizer) -> tuple[int, int]:
    """Assert the assumption-5 token ids on the LIVE tokenizer; return (user, assistant)."""
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    nl = enc("\n")
    assert nl == [NL_ID], f"newline id drift: {nl} != [{NL_ID}]"
    user = enc("user")
    assert len(user) == 1, f"'user' not single-token: {user}"
    assert user[0] == USER_ID_EXPECTED, f"'user' id {user[0]} != {USER_ID_EXPECTED}"
    assistant = enc("assistant")
    assert len(assistant) == 1 and assistant[0] == ASSISTANT_ID_EXPECTED, assistant
    assert tokenizer.convert_tokens_to_ids("<|im_start|>") == IM_START_ID
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == IM_END_ID
    return user[0], assistant[0]


def build_prompt_ids_with_masks(tokenizer, instance: dict, probe: str) -> dict:
    """Tokenize one (instance, probe) PROMPT with exact content/template masks.

    Reconstructs the Qwen2.5 chat template token-by-token —
    ``<|im_start|>role\\n{content}<|im_end|>\\n`` per message (default system
    injected when the instance has none) + the trailing assistant header
    ``<|im_start|>assistant\\n`` — and ASSERTS equality with the tokenizer's own
    ``apply_chat_template(..., add_generation_prompt=True)`` output (fail loud on
    template drift; verified exact on the production tokenizer 2026-07-03).

    Returns a dict with:
      prompt_ids (list[int]),
      content_pos (list[int]) — positions of instance-provided system+user content
        tokens (probe INCLUDED; template-injected default system EXCLUDED;
        assistant prefix-turn content EXCLUDED — plan §3.1 "system+user TEXT"),
      lastk_pos (list[int|None] length 8) — k-th content token from the end,
      trailing template singles: tt_im_end, tt_nl, tt_im_start, tt_assistant, ah_nl.
    """
    from issue594_common import messages_for_instance

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    messages = messages_for_instance(instance, probe)
    ms = list(messages)
    if not ms or ms[0]["role"] != "system":
        ms = [{"role": "system", "content": DEFAULT_QWEN_SYSTEM, "_injected": True}, *ms]
    ids: list[int] = []
    content_pos: list[int] = []
    for m in ms:
        ids += [IM_START_ID, *enc(m["role"]), NL_ID]
        c_ids = enc(m["content"])
        if m["role"] in ("system", "user") and not m.get("_injected", False):
            content_pos.extend(range(len(ids), len(ids) + len(c_ids)))
        ids += [*c_ids, IM_END_ID, NL_ID]
    # positions of the user-turn-closing template block, BEFORE the assistant header
    tt_im_end = len(ids) - 2  # the <|im_end|> closing the probe turn
    tt_nl = len(ids) - 1
    ids += [IM_START_ID]
    tt_im_start = len(ids) - 1
    ids += enc("assistant")
    tt_assistant = len(ids) - 1
    ids += [NL_ID]
    ah_nl = len(ids) - 1  # the last input token — the #658 locked c_C read

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    real = tokenizer(prompt_text, padding=False)["input_ids"]
    if real != ids:
        raise RuntimeError(
            f"[template-reconstruction-assert] manual chat-template build diverges from "
            f"apply_chat_template for {instance['id']} probe={probe[:40]!r} "
            f"(len {len(ids)} vs {len(real)}) — content masks would be wrong; refusing"
        )
    lastk = [content_pos[-k] if len(content_pos) >= k else None for k in range(1, N_LASTK + 1)]
    return {
        "prompt_ids": ids,
        "content_pos": content_pos,
        "lastk_pos": lastk,
        "tt_im_end": tt_im_end,
        "tt_nl": tt_nl,
        "tt_im_start": tt_im_start,
        "tt_assistant": tt_assistant,
        "ah_nl": ah_nl,
    }


def build_full_row(tokenizer, instance: dict, probe: str, answer: str, user_id: int) -> dict | None:
    """One (prompt + answer + 5-token boundary block) teacher-forced row.

    Extends the #810 ``_build_probe_row`` boundary append 2→5 tokens with fed-id
    asserts [151645, 198, 151644, user_id, 198] (the plan §3.2 block). Returns
    None for an empty completion. All positions are PRE-PAD absolute indices.
    """
    pm = build_prompt_ids_with_masks(tokenizer, instance, probe)
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(ans_ids) == 0:
        return None
    p = len(pm["prompt_ids"])
    a = len(ans_ids)
    boundary = [IM_END_ID, NL_ID, IM_START_ID, user_id, NL_ID]
    full_ids = pm["prompt_ids"] + ans_ids + boundary
    fed = full_ids[p + a : p + a + 5]
    assert fed == boundary, f"boundary fed-id mismatch for {instance['id']}: {fed} != {boundary}"
    return {
        **pm,
        "full_ids": full_ids,
        "prompt_len": p,
        "ans_len": a,
        "ans_start": p,  # answer content [p, p+a)
        "b_im_end": p + a,  # boundary singles
        "b_nl": p + a + 1,
        "b_im_start": p + a + 2,
        "b_user": p + a + 3,
        "b_uh_nl": p + a + 4,
    }


def position_slots(ans_len: int) -> tuple[list[int | None], list[int]]:
    """Per-probe (relative-to-answer-start) indices + validity for the 20 position slots.

    head_j (j=0..9): answer-content position j; VALID_OK iff j < ans_len.
    tail_k (k=1..10): end-aligned position ans_len − k; VALID_OK iff in range AND its
    absolute content position ≥ 10 (outside the head window) — a tail slot that
    coincides with a head slot is captured but flagged VALID_DEDUP (plan §3.3
    dedupe-by-absolute-position; the #810-compat gate reduction keeps it, the
    production probe-mean masks it).
    """
    rel: list[int | None] = []
    valid: list[int] = []
    for j in range(N_HEAD):
        if j < ans_len:
            rel.append(j)
            valid.append(VALID_OK)
        else:
            rel.append(None)
            valid.append(VALID_MISSING)
    for k in range(1, N_TAIL + 1):
        pos = ans_len - k
        if pos < 0:
            rel.append(None)
            valid.append(VALID_MISSING)
        else:
            rel.append(pos)
            valid.append(VALID_DEDUP if pos <= N_HEAD - 1 else VALID_OK)
    return rel, valid


# ── sentinel (poll_pipeline contract; pod-side, never task.py) ───────────────


def write_sentinel(kind: str, note: dict, fallback_dir: Path, slug_extra: str = "") -> Path:
    """poll_pipeline.py-conformant sentinel under /workspace/logs (fallback: local dir)."""
    slug = kind.replace(":", "_") + (f"-{slug_extra}" if slug_extra else "")
    log_dir = Path("/workspace/logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / f"issue-920-{slug}-{int(time.time())}.json"
    except OSError:
        fallback_dir.mkdir(parents=True, exist_ok=True)
        target = fallback_dir / f"issue-920-{slug}-sentinel.json"
    dump_json(
        {
            "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
            "kind": kind,
            "version": 1,
            "note": note,
            "ts": int(time.time()),
        },
        target,
    )
    logger.info("wrote sentinel %s", target)
    return target


def reproducibility_metadata() -> dict:
    from issue810_common import reproducibility_metadata as _rm

    return _rm()


def resolve_hf_revision() -> str:
    """The RESOLVED HF dataset-repo revision (commit sha) at fetch time (check (f))."""
    from huggingface_hub import HfApi

    return HfApi().repo_info(HF_DATA_REPO, repo_type="dataset").sha
