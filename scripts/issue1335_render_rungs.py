"""Issue #1335: rung registry + render/datagen for the ablation ladder.

One-factor-per-rung ladder between the strong assistant Q&A map (#825) and the
weak per-character fiction map (#1310). This module is the single source of
truth for the rung set, the per-rung render functions, the render-config
fingerprints ({rung_slug, render_config_hash, code_sha} — the c24 resume
provenance guard), and the teacher-forced (tf) re-render path.

Rungs (plan §4.1/§5):
  r0_qa_full        plain-text Q&A, full answers (M0)          [strong endpoint]
  r1_qa_oneline     same contexts, one-line answers (M1)       [G reference]
  r2_tf / r2_op     renamed responder (Assistant: -> Wren:), tf + on-policy
  r3_persona        + one-line persona description header
  r4_fictionframe   + fiction scene frame, foil asker (Sam:)
  r6_nofoil         #1310 prefill scenes, foils removed
  r7_endpoint       #1310 v3 prefill recipe verbatim           [weak endpoint]
  s1_assistant_label / s2a_familiar / s2b_novel   tf label restorations on r7

Format contract: Q&A rungs (r0-r4) render PLAIN TEXT for BOTH models
(`<asker>: <q>\\n<LABEL>:`); fiction rungs (r6/r7/s*) reproduce the #1310 v3
prefill recipe VERBATIM (instruct header rides the chat template exactly as in
`issue1310_prefill.py` — required for the binding r7 anchor gate). The
base-prime flag (`_BASE_PRIME_OPENERS` 3-turn primes) is uniformly ABSENT from
every fiction-render base cell (r4/r6/r7/s*): the v3 prefill recipe never used
the openers; `assert_base_prime_uniform()` pins this.

Token-id join rule (#1092 BPE seam): capture concatenates STORED prompt +
completion token ids, never re-tokenizes the join; intra-prompt boundaries
(n_prefix_tokens for the v_P arm) are derived from the prompt's offset mapping
at render/gen time (count tokens ENDING inside the prefix text).

CLI:
  uv run python scripts/issue1335_render_rungs.py --write-configs [--data-dir D]
  uv run python scripts/issue1335_render_rungs.py --fetch-questions [--n-questions N]
  uv run python scripts/issue1335_render_rungs.py --tf-rerender --rung s1_assistant_label \
      --model base [--data-dir D]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402
import issue1310_common as c1310  # noqa: E402

SCRIPT = "scripts/issue1335_render_rungs.py"
ISSUE = 1335
BUILD_SEED = 1335  # NEW construction only; the fiction battery keeps BUILD_SEED 1310
GEN_SEED = c1310.GEN_SEED  # 42
FIT_SEED = 0
N_QUESTIONS = 5000

HF_DATA_REPO = c1310.HF_DATA_REPO
HF_PREFIX = "issue1335_ablation_ladder"

# #825 Track-S prompt artifact (reused verbatim; plan §10 Reused data artifacts).
TRACKS_REMOTE = "issue825_userbase_map/raw_completions/track_s/track_s.jsonl"
TRACKS_REV = "deb7a452"  # pinned HF data-repo revision (plan assumption 3)
TRACKS_EXPECT_ROWS = 5000
# Restream fallback (plan assumptions 3/13): on a pinned-artifact miss, the #825
# Track-S prompt set is reproduced by re-streaming lmsys-chat-1m at the build
# revision recorded in track_s_meta.json (dataset_revision @ rev deb7a452) with
# the #825 keep filter (issue779_collect.load_train_contexts: first-turn content,
# non-empty after strip, first 5000 in dataset order). Identity is asserted via
# the prompt-set hash below — sha256(json.dumps(prompts, ensure_ascii=True)) of
# the pinned artifact's ordered "prompt" column (computed 2026-07-15 from
# track_s.jsonl @ deb7a452; 5000 rows).
LMSYS_REPO = "lmsys/lmsys-chat-1m"
LMSYS_REV = "200748d9d3cddcc9d782887541057aca0b18c5da"
TRACKS_PROMPT_SHA256 = "55c5d462ac016d8d794ddc5d557ee741f89ece1cca8bbcbde7529e7be392b42b"
# Bounded-scan cap: keep-rate on lmsys first turns is ~1, so 5000 kept needs
# ~5000 scanned; the cap terminates a pathological 0-keep chain in seconds
# instead of streaming ~1M rows (#1092 tiny-real streaming-probe rule).
TRACKS_RESTREAM_MAX_SCAN = 50_000

QA_LABEL = "Assistant"
RENAMED_LABEL = "Wren"  # the #1310 assistant-adjacent persona
FAMILIAR_NAME = "Sarah"
# "Vexril" (the plan's ungrounded-needs-smoke-test candidate) tokenizes to 3
# tokens on Qwen-2.5-7B; "Xelor" is a rare invented name at 2 tokens (the
# 1-2-token band the plan requires; in-process assert at tf-rerender time).
NOVEL_NAME = "Xelor"
R4_ASKER = "Sam"  # a #1310 foil name, fixed across rows

R0_MAX_TOKENS = 1024  # #825 full-answer convention
R0_STOP = ["\nUser:"]  # plain-text full answer ends when the model opens a new turn
ONELINE_MAX_TOKENS = c1310.SLOT_MAX_TOKENS  # 96
ONELINE_STOP = list(c1310.PREFILL_STOP)  # ["\n"]
CONTEXT_CAP_TOKENS = c1310.CONTEXT_CAP_TOKENS  # 512
CONTEXT_MIN_TOKENS = c1310.CONTEXT_MIN_TOKENS  # 8
DIALOGUE_MIN_TOKENS = c1310.DIALOGUE_MIN_TOKENS  # 4
ROW_MAX_TOKENS = 2048  # #825/#1310 row cap (capture-side drop)
# vLLM engine cap is 4096 (issue1310_prefill.build_engine); leave full answer +
# header margin so no rendered prompt can crash generation (#952 loader rule).
QUESTION_TOKEN_BUDGET = 4096 - R0_MAX_TOKENS - 128

MODEL_IDS = c1310.MODEL_IDS
MODEL_KINDS = c1310.MODEL_KINDS

# ---------------------------------------------------------------------------
# Rung registry (plan §5 config slugs)
# ---------------------------------------------------------------------------

RUNGS: dict[str, dict] = {
    "r0_qa_full": {
        "family": "qa",
        "gen": "full",
        "label": QA_LABEL,
        "header": None,
        "asker": "User",
        "group": "row",
        "tf_source": None,
        "extra_summaries": ("y96", "x_spanmean_nocap"),
    },
    "r1_qa_oneline": {
        "family": "qa",
        "gen": "oneline",
        "label": QA_LABEL,
        "header": None,
        "asker": "User",
        "group": "row",
        "tf_source": None,
    },
    "r2_tf": {
        "family": "qa",
        "gen": "tf",
        "label": RENAMED_LABEL,
        "header": None,
        "asker": "User",
        "group": "row",
        "tf_source": "r1_qa_oneline",
    },
    "r2_op": {
        "family": "qa",
        "gen": "oneline",
        "label": RENAMED_LABEL,
        "header": None,
        "asker": "User",
        "group": "row",
        "tf_source": None,
    },
    "r3_persona": {
        "family": "qa",
        "gen": "oneline",
        "label": RENAMED_LABEL,
        "header": "persona",
        "asker": "User",
        "group": "row",
        "tf_source": None,
    },
    "r4_fictionframe": {
        "family": "qa",
        "gen": "oneline",
        "label": RENAMED_LABEL,
        "header": "scene",
        "asker": R4_ASKER,
        "group": "scenario",
        "tf_source": None,
        "base_prime": False,
    },
    "r6_nofoil": {
        "family": "fiction",
        "gen": "prefill",
        "foils": "none",
        "group": "scene",
        "tf_source": None,
        "base_prime": False,
    },
    "r7_endpoint": {
        "family": "fiction",
        "gen": "prefill",
        "foils": "battery",
        "group": "scene",
        "tf_source": None,
        "base_prime": False,
    },
    "s1_assistant_label": {
        "family": "fiction",
        "gen": "tf",
        "label_override": QA_LABEL,
        "group": "scene",
        "tf_source": "r7_endpoint",
        "base_prime": False,
    },
    "s2a_familiar": {
        "family": "fiction",
        "gen": "tf",
        "label_override": FAMILIAR_NAME,
        "group": "scene",
        "tf_source": "r7_endpoint",
        "base_prime": False,
    },
    "s2b_novel": {
        "family": "fiction",
        "gen": "tf",
        "label_override": NOVEL_NAME,
        "group": "scene",
        "tf_source": "r7_endpoint",
        "base_prime": False,
    },
}
RUNG_ORDER = tuple(RUNGS)
QA_RUNGS = tuple(s for s, c in RUNGS.items() if c["family"] == "qa")
FICTION_RUNGS = tuple(s for s, c in RUNGS.items() if c["family"] == "fiction")
TF_RUNGS = tuple(s for s, c in RUNGS.items() if c["gen"] == "tf")
# Fiction-RENDER rungs (base-prime pin scope): every rung whose base cells go
# through the fiction scene-header render path (plan §4.2 base-prime pin).
FICTION_RENDER_RUNGS = ("r4_fictionframe", *FICTION_RUNGS)


def assert_base_prime_uniform() -> bool:
    """The base-prime-opener flag must be IDENTICAL across all fiction-render
    base cells (r4/r6/r7/s*) so opener presence never rides a ladder delta."""
    flags = {slug: bool(RUNGS[slug].get("base_prime", False)) for slug in FICTION_RENDER_RUNGS}
    assert len(set(flags.values())) == 1, f"base_prime flag differs across rungs: {flags}"
    return next(iter(flags.values()))


# ---------------------------------------------------------------------------
# Fingerprints (c24 resume provenance)
# ---------------------------------------------------------------------------


def rung_render_config(slug: str) -> dict:
    """Canonical, hashable render config for one rung (everything that changes
    the rendered rows: template constants, seeds, caps, labels, sampling)."""
    cfg = RUNGS[slug]
    base = {
        "issue": ISSUE,
        "rung_slug": slug,
        "rung_cfg": {k: v for k, v in cfg.items()},
        "model_ids": dict(MODEL_IDS),
        "gen_seed": GEN_SEED,
        "temperature": c1310.GEN_TEMPERATURE,
        "top_p": c1310.GEN_TOP_P,
        "context_cap_tokens": CONTEXT_CAP_TOKENS,
        "context_min_tokens": CONTEXT_MIN_TOKENS,
        "dialogue_min_tokens": DIALOGUE_MIN_TOKENS,
        "row_max_tokens": ROW_MAX_TOKENS,
    }
    if cfg["family"] == "qa":
        base.update(
            {
                "n_questions": N_QUESTIONS,
                "question_source": {"remote": TRACKS_REMOTE, "revision": TRACKS_REV},
                "question_token_budget": QUESTION_TOKEN_BUDGET,
                "max_tokens": R0_MAX_TOKENS if cfg["gen"] == "full" else ONELINE_MAX_TOKENS,
                "stop": R0_STOP if cfg["gen"] == "full" else ONELINE_STOP,
                "persona_desc": c1310.PERSONAS[RENAMED_LABEL] if cfg["header"] else None,
            }
        )
        if cfg["header"] == "scene":
            base["battery_seed"] = c1310.BUILD_SEED
            base["n_scenarios"] = c1310.N_PROMPTS_PER_PERSONA
    else:
        base.update(
            {
                "battery_seed": c1310.BUILD_SEED,
                "n_scenarios": c1310.N_PROMPTS_PER_PERSONA,
                "slots": c1310.PREFILL_SLOTS,
                "max_tokens": ONELINE_MAX_TOKENS,
                "stop": ONELINE_STOP,
                "personas": dict(c1310.PERSONAS),
                "foil_names": list(c1310.FOIL_NAMES),
                "foil_lines": list(c1310._FOIL_LINES),
            }
        )
    return base


def render_config_hash(slug: str) -> str:
    """Deterministic sha256 over the canonical render config JSON."""
    blob = json.dumps(rung_render_config(slug), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fingerprint(slug: str) -> dict:
    """{rung_slug, render_config_hash, code_sha} — stamped on every gen record,
    shard sidecar, and fit JSON; resume skips a cell ONLY on exact match."""
    return {
        "rung_slug": slug,
        "render_config_hash": render_config_hash(slug),
        "code_sha": common.git_commit(),
    }


def fingerprint_matches(sidecar: dict, slug: str, *, require_sha: bool = True) -> bool:
    """c24 predicate. RESUME-SKIP decisions require the FULL fingerprint match
    (rung_slug + render_config_hash + code_sha — plan §8). CONSUME checks
    (fitting an existing store) pass require_sha=False: the render-config hash
    pins the store's DATA identity, and a code-sha-only difference (e.g. a fix
    commit between capture and fit in one run) is tolerated with a caller-side
    warning rather than forcing a recapture of identical renders."""
    want = fingerprint(slug)
    keys = ("rung_slug", "render_config_hash") + (("code_sha",) if require_sha else ())
    return all(sidecar.get(k) == want[k] for k in keys)


# ---------------------------------------------------------------------------
# Q&A renders (plain text, both models)
# ---------------------------------------------------------------------------


def persona_header_text(label: str) -> str:
    """One-line #1310 persona description header (the prefill_header clause)."""
    return f"{label} is {c1310.PERSONAS[RENAMED_LABEL]}.\n\n"


def fiction_header_text(
    scenario: dict, label: str, desc: str, foils: list[str], model_kind: str
) -> str:
    """Foils-parameterized scene header; byte-identical to
    ``issue1310_common.prefill_header`` when foils == foils_for_scene(...) and
    label/desc are the persona's own (asserted by tests)."""
    setting = c1310._cap(scenario["setting"])
    situation = c1310._cap(scenario["situation"])
    foil_list = ", ".join(foils)
    if model_kind == "base":
        head = (
            f"The following is a dialogue scene in script format. "
            f"Setting: {setting}. Situation: {situation}. "
            f"{label} is {desc}. "
        )
        if foils:
            head += f"Also present: {foil_list}. "
        head += "Each line is `Name: what they say`.\n\n"
        return head
    if model_kind == "instruct":
        char_lines = f"- {label}: {desc}\n" + "".join(
            f"- {f}: another person caught up in the same situation\n" for f in foils
        )
        return (
            f"Write a dialogue scene in strict SCRIPT format. "
            f"Setting: {setting}. Situation: {situation}.\n\n"
            f"Characters:\n{char_lines}\n"
            f"Format EVERY line as `Name: what they say` — one speaker turn per line, "
            f"no narration, no stage directions, no blank lines, no quotation marks. "
            f"{label} speaks in {label}'s own voice."
        )
    raise ValueError(f"unknown model_kind {model_kind!r}")


def qa_render(slug: str, question: str, scenario: dict | None = None) -> tuple[str, str]:
    """Render one Q&A-rung prompt. Returns (prompt, prefix_text).

    prompt = <prefix_text><asker>: <q>\\n<LABEL>:  (plain text, both models);
    prefix_text = everything before the final query turn (the v_P arm span);
    empty for the header-less rungs r0/r1/r2 (structurally degenerate arm).
    """
    cfg = RUNGS[slug]
    assert cfg["family"] == "qa", slug
    label = cfg["label"]
    if cfg["header"] is None:
        prefix = ""
    elif cfg["header"] == "persona":
        prefix = persona_header_text(label)
    elif cfg["header"] == "scene":
        assert scenario is not None, f"{slug} needs a scenario"
        prefix = fiction_header_text(
            scenario, label, c1310.PERSONAS[RENAMED_LABEL], [cfg["asker"]], "base"
        )
    else:
        raise ValueError(f"unknown header {cfg['header']!r}")
    prompt = f"{prefix}{cfg['asker']}: {question}\n{label}:"
    return prompt, prefix


# ---------------------------------------------------------------------------
# Fiction prefill renders (foils-parameterized, #1310-verbatim at foils=battery)
# ---------------------------------------------------------------------------


def foils_for_rung(slug: str, scenario_id: str) -> list[str]:
    """r7/s* -> the #1310 battery foils; r6 -> none."""
    cfg = RUNGS[slug]
    src = RUNGS[cfg["tf_source"]] if cfg.get("tf_source") else cfg
    if src["foils"] == "battery":
        return c1310.foils_for_scene(scenario_id)
    assert src["foils"] == "none", src
    return []


def canned_foil_turn(scenario_id: str, slot: int, foils: list[str]) -> str:
    """Foils-param twin of c1310.canned_foil_turn (byte-identical on battery foils)."""
    assert foils, "no canned foil turn without foils"
    foil = foils[slot % len(foils)]
    line = c1310._FOIL_LINES[slot % len(c1310._FOIL_LINES)]
    return f"{foil}: {line}"


def fiction_body_slot0(scenario_id: str, foils: list[str]) -> str:
    """Opening body: canned foil turn (r7 shape) or empty (r6, no foils)."""
    if not foils:
        return ""
    return canned_foil_turn(scenario_id, 0, foils) + "\n"


def fiction_advance_body(
    body: str, label: str, completion: str, scenario_id: str, next_slot: int, foils: list[str]
) -> str:
    """Append the character's generated turn (+ next canned foil turn if foils)."""
    if foils:
        return f"{body}{label}:{completion}\n{canned_foil_turn(scenario_id, next_slot, foils)}\n"
    return f"{body}{label}:{completion}\n"


def fiction_prefix(
    tokenizer,
    scenario: dict,
    persona: str,
    model_kind: str,
    body: str,
    foils: list[str],
    label_override: str | None = None,
) -> str:
    """Full prefill prompt (header [+ chat template on instruct] + body + cue).

    Byte-identical to ``issue1310_prefill.build_prefix`` when foils == the
    battery foils and label_override is None (the r7 endpoint-parity contract).
    """
    label = label_override or persona
    header = fiction_header_text(scenario, label, c1310.PERSONAS[persona], foils, model_kind)
    if model_kind == "instruct":
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": header}], tokenize=False, add_generation_prompt=True
        )
        return f"{templated}{body}{label}:"
    return f"{header}{body}{label}:"


def fiction_prefix_text(
    prompt: str, scenario_id: str, slot: int, foils: list[str], label: str
) -> str:
    """v_P prefix text for a fiction row: everything before the final query/foil
    line (r7: drop the trailing canned foil turn; r6: everything before the cue)."""
    cue = f"{label}:"
    assert prompt.endswith(cue), (scenario_id, slot, "prompt does not end with the cue")
    before_cue = prompt[: -len(cue)]
    if foils:
        last_foil = canned_foil_turn(scenario_id, slot, foils) + "\n"
        assert before_cue.endswith(last_foil), (scenario_id, slot, "missing trailing foil turn")
        return before_cue[: -len(last_foil)]
    return before_cue


# ---------------------------------------------------------------------------
# Prefix-token counting (offset-mapping based; #1092 seam rule)
# ---------------------------------------------------------------------------


def count_prefix_tokens(
    tokenizer, prompt: str, prefix_text: str, prompt_token_ids: list[int]
) -> int:
    """Number of prompt tokens ENDING inside prefix_text (offset-mapping based).

    Asserts the HF fast tokenization of the whole prompt reproduces the stored
    prompt_token_ids (fails loud on any generation-vs-capture tokenizer drift).
    """
    assert prompt.startswith(prefix_text), "prefix_text must be a string prefix of prompt"
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    assert ids == list(prompt_token_ids), (
        f"tokenizer drift: HF re-tokenization ({len(ids)} toks) != stored prompt ids "
        f"({len(prompt_token_ids)} toks)"
    )
    plen = len(prefix_text)
    return sum(1 for s, e in enc["offset_mapping"] if e <= plen and e > s)


# ---------------------------------------------------------------------------
# Question loading (#825 Track-S prompts, reused verbatim)
# ---------------------------------------------------------------------------


def _select_track_s_prompts(rows, n: int, max_scan: int = TRACKS_RESTREAM_MAX_SCAN) -> list[str]:
    """#825 Track-S keep filter (issue779_collect.load_train_contexts, lmsys leg).

    Takes each row's first `conversation` turn content (`content` else `value`),
    keeps non-empty stripped strings, stops at the first n in dataset order.
    Fail-loud on a short yield within max_scan scanned rows (a mis-shaped filter
    must never silently under-fill; #1092). Returns the ordered prompt list.
    """
    prompts: list[str] = []
    n_scanned = n_dropped = 0
    for row in rows:
        n_scanned += 1
        val = row.get("conversation")
        p = None
        if isinstance(val, list) and val and isinstance(val[0], dict):
            p = val[0].get("content") or val[0].get("value")
        elif isinstance(val, str):
            p = val
        if p and isinstance(p, str) and len(p.strip()) > 0:
            prompts.append(p.strip())
        else:
            n_dropped += 1
        if len(prompts) >= n:
            break
        if n_scanned >= max_scan:
            break
    print(
        f"[i1335-render] restream select done: kept={len(prompts)} scanned={n_scanned} "
        f"dropped_empty_first_turn={n_dropped}"
    )
    if len(prompts) < n:
        raise RuntimeError(
            f"Track-S restream yielded only {len(prompts)}/{n} prompts after scanning "
            f"{n_scanned} rows (cap {max_scan}) — filter/revision mismatch, not padded."
        )
    return prompts


def _prompt_set_sha256(prompts: list[str]) -> str:
    """Order-sensitive content hash of the prompt set (unambiguous JSON encoding)."""
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=True).encode("utf-8")).hexdigest()


def _iter_lmsys_rows(revision: str):
    """Stream lmsys-chat-1m train rows at the pinned revision (network boundary)."""
    from datasets import load_dataset

    return load_dataset(LMSYS_REPO, split="train", streaming=True, revision=revision)


def _restream_track_s(
    dest: Path, rows=None, expected_sha: str | None = None, n: int | None = None
) -> None:
    """Reproduce the #825 Track-S prompt set by re-streaming lmsys (fallback).

    Verified by row count (default TRACKS_EXPECT_ROWS) + the pinned prompt-set
    content hash (default TRACKS_PROMPT_SHA256; both resolved at CALL time so
    fixture-scale tests can override); any mismatch raises (fail-loud — never a
    silently different corpus). Writes load_questions-compatible JSONL rows
    ({prompt_idx, prompt}).
    """
    import gc

    expected_sha = TRACKS_PROMPT_SHA256 if expected_sha is None else expected_sha
    n = TRACKS_EXPECT_ROWS if n is None else n
    rows_iter = rows if rows is not None else _iter_lmsys_rows(LMSYS_REV)
    try:
        prompts = _select_track_s_prompts(rows_iter, n)
    finally:
        # Release the streaming IterableDataset while the interpreter is healthy
        # (a survivor at shutdown SIGABRTs rc=134 in the pinned datasets env; #952).
        del rows_iter
        gc.collect()
    sha = _prompt_set_sha256(prompts)
    if sha != expected_sha:
        raise RuntimeError(
            f"Track-S restream prompt-set hash mismatch: got {sha}, expected "
            f"{expected_sha} (lmsys rev {LMSYS_REV[:12]}) — the restream did NOT "
            "reproduce the pinned #825 prompt set; halting rather than running a "
            "silently different corpus."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"prompt_idx": i, "prompt": p}, ensure_ascii=False) + "\n")
    tmp.replace(dest)
    print(
        f"[i1335-render] restreamed Track-S prompts (lmsys rev {LMSYS_REV[:12]}, "
        f"n={len(prompts)}, prompt_set_sha256={sha[:12]}) -> {dest}"
    )


def _run_restream_child(dest: Path) -> int:
    """Spawn the restream in a CHILD process (network boundary); returns its rc."""
    import subprocess

    cmd = [sys.executable, str(Path(__file__).resolve()), "--restream-track-s", str(dest)]
    return subprocess.run(cmd, env={**os.environ}, check=False).returncode


def _restream_track_s_subprocess(dest: Path, run_child=None) -> None:
    """Subprocess-isolated restream, routed on the ARTIFACT (rc is secondary).

    The pinned datasets/pyarrow env can SIGABRT at interpreter SHUTDOWN after
    all restream work completed and the artifact was written (rc=134;
    gotchas.md HF-datasets shutdown-abort class — reproduced on the 2026-07-15
    live smoke despite the in-process del+gc release). Isolating the stream in
    a child keeps that abort out of the CALLER (which may be the GPU gen
    process); the parent independently re-verifies row count + the pinned
    prompt-set hash from the artifact, so a tolerated nonzero rc can never
    accept wrong data (fail-loud preserved).
    """
    runner = _run_restream_child if run_child is None else run_child
    rc = runner(dest)
    if not dest.exists():
        raise RuntimeError(f"Track-S restream child rc={rc} and no artifact at {dest}")
    prompts = [json.loads(line)["prompt"] for line in dest.open(encoding="utf-8") if line.strip()]
    sha = _prompt_set_sha256(prompts)
    if len(prompts) != TRACKS_EXPECT_ROWS or sha != TRACKS_PROMPT_SHA256:
        raise RuntimeError(
            f"Track-S restream artifact failed parent verification (child rc={rc}): "
            f"n={len(prompts)} (want {TRACKS_EXPECT_ROWS}), prompt_set_sha256={sha[:12]} "
            f"(want {TRACKS_PROMPT_SHA256[:12]})"
        )
    if rc != 0:
        print(
            f"[i1335-render] restream child rc={rc} TOLERATED — artifact verified "
            "(row count + prompt-set hash; HF-datasets shutdown-abort class, gotchas.md)"
        )


def _fetch_track_s(dest: Path) -> None:
    """Stage the pinned #825 Track-S artifact; on a miss, auto-restream lmsys.

    Order: pinned HF revision -> main -> lmsys restream at the #825 build
    revision (plan assumptions 3/13). Only when the restream ALSO fails does
    staging halt fail-loud.
    """
    from huggingface_hub import hf_hub_download

    last_err: Exception | None = None
    for rev in (TRACKS_REV, None):
        try:
            cached = hf_hub_download(HF_DATA_REPO, TRACKS_REMOTE, repo_type="dataset", revision=rev)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached, dest)
            print(f"[i1335-render] staged Track-S prompts (revision={rev or 'main'}) -> {dest}")
            return
        except Exception as e:
            last_err = e
            print(f"[i1335-render] Track-S fetch at revision={rev!r} failed: {e!r}")
    print(
        "[i1335-render] pinned Track-S artifact unavailable — auto-restreaming "
        f"lmsys-chat-1m @ {LMSYS_REV[:12]} (plan assumption 3 fallback)"
    )
    try:
        _restream_track_s_subprocess(dest)
    except Exception as e2:
        raise RuntimeError(
            f"Track-S prompt artifact unavailable at {HF_DATA_REPO}/{TRACKS_REMOTE} "
            f"(rev {TRACKS_REV} and main; hf error: {last_err!r}) AND the lmsys "
            f"restream fallback failed ({e2!r}). No recovery path — halting."
        ) from e2


def load_questions(data_dir: Path, n: int = N_QUESTIONS, tokenizer=None) -> tuple[list[dict], dict]:
    """Load the #825 Track-S question set (real lmsys user prompts).

    Returns (rows, meta): rows = [{"q_idx", "question"}] with a load-time
    length filter (#952 loader rule — the LONGEST render must fit the vLLM
    engine cap; drops recorded digest-only, never padded). Row text is
    real-world-corpus content: never print it.
    """
    local = data_dir / "track_s.jsonl"
    if not local.exists():
        _fetch_track_s(local)
    raw = []
    with local.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw.append(json.loads(line))
    assert len(raw) == TRACKS_EXPECT_ROWS, f"{local}: {len(raw)} rows != {TRACKS_EXPECT_ROWS}"
    if tokenizer is None:
        tokenizer = common.get_tokenizer(MODEL_IDS["base"])
    rows, dropped = [], []
    for i, r in enumerate(raw):
        q = r["prompt"]
        n_tok = len(tokenizer(q, add_special_tokens=False)["input_ids"])
        if not q.strip() or n_tok > QUESTION_TOKEN_BUDGET:
            dropped.append({"q_idx": i, "n_tokens": n_tok})
            continue
        rows.append({"q_idx": i, "question": q})
    rows = rows[:n] if n else rows
    meta = {
        "n_raw": len(raw),
        "n_kept": len(rows),
        "n_dropped_overlong_or_empty": len(dropped),
        "dropped_digest": dropped[:50],
        "question_token_budget": QUESTION_TOKEN_BUDGET,
        "source_sha256": common.sha256_file(local),
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Teacher-forced re-renders (r2_tf, s1, s2a, s2b) — capture-only, no generation
# ---------------------------------------------------------------------------


def gen_path(data_dir: Path, slug: str, model_kind: str) -> Path:
    return data_dir / "generation" / slug / f"{model_kind}_gen.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    """Newline-split JSONL read (splitlines shreds U+2028-bearing rows)."""
    assert path.exists(), f"missing input: {path}"
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_gen_jsonl(path: Path, records: list[dict]) -> None:
    """Atomic JSONL write (tmp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _tf_record(src: dict, slug: str, prompt: str, prefix_text: str, tokenizer, fp: dict) -> dict:
    """One tf re-render record: fresh whole-prompt HF tokenization + the STORED
    completion ids from the source on-policy record (seam-safe join)."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(enc["input_ids"])
    plen = len(prefix_text)
    n_prefix = sum(1 for s, e in enc["offset_mapping"] if e <= plen and e > s)
    rec = dict(src)
    rec.update(
        {
            "rung": slug,
            "row_id": f"{slug}:{src['row_id'].split(':', 1)[-1]}"
            if src["row_id"].startswith(src["rung"])
            else src["row_id"],
            "prompt": prompt,
            "prompt_token_ids": prompt_ids,
            "n_prompt_tokens": len(prompt_ids),
            "n_prefix_tokens": n_prefix,
            "provenance": "tf-rerender",
            "tf_source_rung": src["rung"],
            "tf_source_row_id": src["row_id"],
            **fp,
        }
    )
    return rec


def tf_rerender(slug: str, model_kind: str, data_dir: Path) -> Path:
    """Re-render an existing rung's on-policy rows under a swapped label and
    write capture-ready gen records (completion ids copied verbatim)."""
    cfg = RUNGS[slug]
    assert cfg["gen"] == "tf", slug
    src_slug = cfg["tf_source"]
    src_records = _read_jsonl(gen_path(data_dir, src_slug, model_kind))
    tokenizer = common.get_tokenizer(MODEL_IDS[model_kind])
    fp = fingerprint(slug)
    out: list[dict] = []

    if cfg["family"] == "qa":  # r2_tf: relabel the r1 rows Assistant: -> Wren:
        for src in src_records:
            prompt, prefix_text = qa_render(slug, src["question"])
            out.append(_tf_record(src, slug, prompt, prefix_text, tokenizer, fp))
    else:  # s1/s2a/s2b: rebuild the r7 prefill prompts with the override label
        override = cfg["label_override"]
        n_name = len(tokenizer(override, add_special_tokens=False)["input_ids"])
        assert 1 <= n_name <= 2, (
            f"{slug}: override label {override!r} tokenizes to {n_name} tokens "
            "(plan requires 1-2, token-length-matched)"
        )
        by_scene: dict[tuple[str, str], list[dict]] = {}
        for r in src_records:
            by_scene.setdefault((r["scenario_id"], r["persona"]), []).append(r)
        for (sc_id, persona), recs in sorted(by_scene.items()):
            recs = sorted(recs, key=lambda r: r["slot"])
            scenario = {
                "scenario_id": sc_id,
                "setting": recs[0]["setting"],
                "situation": recs[0]["situation"],
            }
            foils = foils_for_rung(slug, sc_id)
            body = fiction_body_slot0(sc_id, foils)
            body_ovr = body
            for rec in recs:
                slot = rec["slot"]
                # Faithfulness self-check: the no-override rebuild reproduces
                # the STORED prompt byte-for-byte before we emit the override.
                orig = fiction_prefix(tokenizer, scenario, persona, model_kind, body, foils)
                assert orig == rec["prompt"], (
                    f"{slug}: rebuilt r7 prompt != stored prompt for "
                    f"{sc_id}:{persona}:t{slot} — reconstruction unfaithful"
                )
                new_prompt = fiction_prefix(
                    tokenizer, scenario, persona, model_kind, body_ovr, foils, override
                )
                prefix_text = fiction_prefix_text(new_prompt, sc_id, slot, foils, override)
                out.append(_tf_record(rec, slug, new_prompt, prefix_text, tokenizer, fp))
                body = fiction_advance_body(
                    body, persona, rec["completion"], sc_id, slot + 1, foils
                )
                body_ovr = fiction_advance_body(
                    body_ovr, override, rec["completion"], sc_id, slot + 1, foils
                )

    dest = gen_path(data_dir, slug, model_kind)
    write_gen_jsonl(dest, out)
    c1310.write_json(
        dest.with_name(f"{model_kind}_gen_manifest.json"),
        {
            "metadata": common.metadata(SCRIPT, GEN_SEED, len(out)),
            **fp,
            "model_kind": model_kind,
            "tf_source": src_slug,
            "n_records": len(out),
            "jsonl_sha256": common.sha256_file(dest),
        },
    )
    print(f"[i1335-render] tf-rerender {slug}/{model_kind}: {len(out)} records -> {dest}")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_configs(data_dir: Path) -> Path:
    """P0: persist all rung render configs + hashes (the fingerprint source)."""
    assert_base_prime_uniform()
    battery = c1310.build_scenario_battery()
    payload = {
        "metadata": common.metadata(SCRIPT, BUILD_SEED, len(RUNGS)),
        "code_sha": common.git_commit(),
        "rungs": {slug: rung_render_config(slug) for slug in RUNG_ORDER},
        "render_config_hashes": {slug: render_config_hash(slug) for slug in RUNG_ORDER},
        "battery": {
            "seed": c1310.BUILD_SEED,
            "n": len(battery),
            "sha256": hashlib.sha256(
                json.dumps(battery, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
        },
        "base_prime_uniform": assert_base_prime_uniform(),
    }
    out = data_dir / "render_configs.json"
    c1310.write_json(out, payload)
    print(f"[i1335-render] wrote {out} ({len(RUNGS)} rungs)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1335"))
    ap.add_argument("--write-configs", action="store_true")
    ap.add_argument("--fetch-questions", action="store_true")
    ap.add_argument("--n-questions", type=int, default=N_QUESTIONS)
    ap.add_argument("--tf-rerender", action="store_true")
    ap.add_argument("--rung", type=str, default=None, choices=[*TF_RUNGS, None])
    ap.add_argument("--model", type=str, default=None, choices=[*MODEL_KINDS, None])
    ap.add_argument(
        "--restream-track-s",
        type=Path,
        default=None,
        metavar="DEST",
        help="child mode: restream the #825 Track-S prompt set from lmsys to DEST "
        "(subprocess-isolated by _fetch_track_s; hash-verified in-process AND by the parent)",
    )
    args = ap.parse_args()
    if args.restream_track_s is not None:
        _restream_track_s(args.restream_track_s)
        return 0
    print("[phase=p0_render] rung render/datagen")
    did = False
    if args.write_configs:
        write_configs(args.data_dir)
        did = True
    if args.fetch_questions:
        rows, meta = load_questions(args.data_dir, args.n_questions)
        c1310.write_json(args.data_dir / "questions_meta.json", meta)
        print(f"[i1335-render] questions ready: kept {len(rows)} (meta -> questions_meta.json)")
        did = True
    if args.tf_rerender:
        assert args.rung and args.model, "--tf-rerender needs --rung and --model"
        tf_rerender(args.rung, args.model, args.data_dir)
        did = True
    assert did, "no action requested (use --write-configs / --fetch-questions / --tf-rerender)"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
