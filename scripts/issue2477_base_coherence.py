"""#2477 — base-model coherence under the chat template (paper base-row format check).

Driver for the approved plan (tasks/running/2477/plans/plan.md, v4). Phases:

- ``inventory``   Plan A1-A3 + A5 pre-parity: scoped RECURSIVE HF listings of the four
                  issue-lineage roots, per-file classification manifest, format table.
- ``sample``      Plan A4 (+ A5 bank-parity gate when inventory surfaced a candidate):
                  seed-2477 draws (200 chat prompts, 125 matched raw pairs), full-corpus
                  structural scan (plan assumption 9), manifest write + HF upload.
- ``gen``         Plan C2-C4 (POD-SIDE ONLY): vLLM two-arm base generation over the
                  sampled 200 prompts, per-arm checkpointing, HF upload + verify, sentinel.
- ``judge-pilot`` Plan B4: ``judge_pilot_gate`` over all 5 arms (260 draws, forced batch).
- ``judge-wave``  Plan B3 (``--smoke``: 5-item live forced-batch probe) / B5 (production
                  wave, 850 items x 5 draws, Batch-API pinned via ``threshold_base=0``).
- ``aggregate``   Plan B6: per-arm stats, bootstrap/Wilson CIs, paired deltas, verdict.
- ``figures``     Plan B7: hero ``coherence_by_arm`` (arm means + bootstrap CIs +
                  per-item strip) + exploratory dump (per-arm item-mean histograms,
                  per-depth lines for the raw arms, cap-hit bar where recorded,
                  distinct-3gram vs judge-score scatter, paired-delta histogram),
                  paper-plots conventions; ``--smoke`` renders from the synthetic
                  aggregate fixture into /tmp scratch (no canonical writes).

Idempotency (round 2): every phase carries an entry guard keyed on its primary output
(gen: ``gen_done.json`` sentinel, checked BEFORE the torch import; judge-pilot: a PASSED
``pilot_report.json`` — a failed pilot re-runs by design; judge-wave: the complete 5-arm
``judge_raw_*.json`` set covering this manifest's item ids). A completed phase SKIPs loud;
``--force`` re-runs. Judge phases run on any machine: ``_fresh_rows`` stages the fresh
completions from the plan-declared HF dest when the pod-local mirror is absent.

Condition sets (plan v5 C0, follow-up `decoding-sensitivity`): ``--condition-set
{parent,decoding-sensitivity}`` routes gen/judge/aggregate/figures through the
``CONDITION_SETS`` registry. Default ``parent`` preserves the parent round's behavior
unchanged; ``decoding-sensitivity`` runs the 4-condition temperature ablation
(chat/bare renders x temperatures {0.7, 0.0}) with set-distinct out roots, Hub prefixes
(``.../generation/decoding_sensitivity/``, ``judge_raw/decoding_sensitivity/``), judge
caches, verdict path (``decoding-sensitivity/coherence_verdict.json``, §3 lattice
render-and-sampling / render-driven) and ``decoding_sensitivity_``-prefixed figures.

Content hygiene: prompts/responses are real LMSYS/WildChat-lineage text — this driver
never PRINTS text fields (digests, counts and shas only); texts live only in artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) + API keys BEFORE any heavy import

# vLLM v1 EngineCore dies silently under fork() when the parent touched
# transformers before LLM() (gotchas.md, #628) — the gen phase loads the
# instruct tokenizer before the engine, so pin spawn at module top.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

REPO_DATA = "superkaiba1/explore-persona-space-data"
EXPERIMENT = "issue2477_base_coherence"
MODEL_BASE = "Qwen/Qwen2.5-7B"
MODEL_INSTRUCT = "Qwen/Qwen2.5-7B-Instruct"
GEN_SEED = 42  # track_s parity (plan §10 seeds)
SAMPLE_SEED_DEFAULT = 2477
N_CHAT = 200
N_RAW_PAIRS = 125
JUDGE_MAX_TOKENS = 1024
N_DRAWS_PROD = 5
N_DRAWS_PILOT = 2
PILOT_TARGET_TOTAL_DRAWS = 260  # satisfiability-guard minimum at 5 arms / 2 draws / 2%
VERDICT_FLOOR = 0.80
COHERENT_THRESHOLD = 50.0
N_BOOT = 10_000

# Plan A2 roots — scoped recursive listings only (full-repo listing banned on ~1M files).
ROOTS = [
    "issue825_userbase_map/raw_completions",
    "issue2061_sae_predictability",
    "issue1902_stage_map",
    "issue1336_rlvr_ladder/raw_completions",
]

TRACK_S_JSONL = "issue825_userbase_map/raw_completions/track_s/track_s.jsonl"
TRACK_S_META = "issue825_userbase_map/raw_completions/track_s/track_s_meta.json"
ARMG_ROOT = "issue825_userbase_map/raw_completions/turn_dynamics/armG"

ARM_NAMES = (
    "arm_instruct_chat",
    "arm_base_chat",
    "arm_base_bare",
    "arm_base_rawmt",
    "arm_instruct_rawmt",
)

# Plan §4 B2 rubric, VERBATIM ({question}/{answer} slots per judge_graded.format_user_msg;
# the module appends its own JSON output wrapper to the system prompt).
RUBRIC = """You are given a user message and a model's response. Rate ONLY the COHERENCE and FLUENCY of the response on a 0-100 scale.

[QUESTION START]
{question}
[QUESTION END]

[ANSWER START]
{answer}
[ANSWER END]

0 means totally incoherent: gibberish, endless repetition, random topic or language jumps, no recognizable engagement with the user message.
50 means somewhat coherent: recognizable on-topic prose with notable problems (local repetition, drift, internal contradictions).
100 means totally coherent: fluent, on-topic, internally consistent from start to end.

Judge ONLY coherence and fluency. Do NOT judge helpfulness, correctness, safety, formatting, or style.
If the response ends abruptly mid-sentence (a length cap), judge the coherence of the text that is present; do not penalize the cut itself.
Gibberish, repetition loops, or off-topic text are NOT reasons to refuse: rate such text low.
Only if the response is completely empty, return the literal token REFUSAL instead of a number."""


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval_results" / "issue_2477"
STAGE_DIR = REPO_ROOT / "data" / "issue_2477" / "hf_dl"

# C0 (plan v5 §4): condition-set registry. Row = (config_slug, render, temperature, stop).
# "parent" reproduces the parent round's phase_gen literals VERBATIM (temperature 1.0) so the
# default --condition-set parent path is behavior-identical to the parent round;
# "decoding-sensitivity" is the 4-condition temperature ablation (chat/bare x {0.7, 0.0};
# 0.7 = Qwen2.5-7B-Instruct generation_config default, 0.0 = greedy bound — plan v5 §11).
CONDITION_SETS: dict[str, tuple[tuple[str, str, float, tuple[str, ...]], ...]] = {
    "parent": (
        ("base_chat", "chat", 1.0, ("<|im_end|>",)),
        ("base_bare", "bare", 1.0, ("\nUser:", "\n\nUser:")),
    ),
    "decoding-sensitivity": (
        ("base_chat_t07", "chat", 0.7, ("<|im_end|>",)),
        ("base_chat_t00", "chat", 0.0, ("<|im_end|>",)),
        ("base_bare_t07", "bare", 0.7, ("\nUser:", "\n\nUser:")),
        ("base_bare_t00", "bare", 0.0, ("\nUser:", "\n\nUser:")),
    ),
}
DECSENS = "decoding-sensitivity"
DECSENS_ARM_NAMES = tuple(f"arm_{slug}" for slug, _r, _t, _s in CONDITION_SETS[DECSENS])
DECSENS_EVAL_DIR = EVAL_DIR / "decoding-sensitivity"
DECSENS_HUB_GEN_PREFIX = f"{EXPERIMENT}/raw_completions/generation/decoding_sensitivity"
DECSENS_HUB_JUDGE_PREFIX = f"{EXPERIMENT}/judge_raw/decoding_sensitivity"
# 4-arm pilot satisfiability minimum: 4 arms x 2 draws x ceil(51/2)=26 items (plan v5 §7 G2').
DECSENS_PILOT_TARGET_TOTAL_DRAWS = 208
# Registered cross-temperature contrasts (plan v5 §3): (key, fresh arm, parent comparator arm).
DECSENS_PAIR_SPECS = (
    ("chat_t07_minus_chat_t10", "arm_base_chat_t07", "arm_base_chat"),
    ("chat_t00_minus_chat_t10", "arm_base_chat_t00", "arm_base_chat"),
    ("bare_t07_minus_bare_t10", "arm_base_bare_t07", "arm_base_bare"),
    ("bare_t00_minus_bare_t10", "arm_base_bare_t00", "arm_base_bare"),
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _meta(phase: str) -> dict:
    prov = git_provenance(cwd=REPO_ROOT)
    out = as_metadata_dict(prov, phase=phase)
    out["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration — NEVER splitlines() (U+2028/NEL shredding, gotchas.md)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n").strip("\r")
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_committed_json(rel: str) -> dict:
    """Read a repo-committed JSON: filesystem first, else the git blob at HEAD.

    The sparse worktree may not check out eval_results/issue_825; the blob is
    still in the tree at HEAD (branched from main), so `git show HEAD:rel`
    recovers it. Fail loud when neither resolves.
    """
    p = REPO_ROOT / rel
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{rel}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(f"{rel}: not on filesystem and not in HEAD tree: {proc.stderr}")
    return json.loads(proc.stdout)


def _distinct_3gram_rate(texts: list[str]) -> float:
    # Verbatim reimplementation of scripts/issue825_gen_conversations.py:324
    # (_distinct_3gram_rate). Reimplemented, not imported: importing the parent
    # script module would execute its module top level (plan §10 item (k)).
    total = 0
    distinct: set[tuple[str, ...]] = set()
    for text in texts:
        words = text.split()
        for j in range(len(words) - 2):
            total += 1
            distinct.add(tuple(words[j : j + 3]))
    return (len(distinct) / total) if total else 0.0


def _strip_header(text: str, header: str) -> tuple[str, bool]:
    """Strip ONE leading literal role header (plan B1 render-artifact substitution)."""
    if text.startswith(header):
        return text[len(header) :], True
    return text, False


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _boot_ci_mean(vals: list[float], n_boot: int = N_BOOT) -> tuple[float, float]:
    """10k-draw percentile bootstrap CI of the mean over items; fresh rng(0) per CI
    (plan B6 'rng(0)'; per-CI seeding keeps each interval reproducible in isolation)."""
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# ---------------------------------------------------------------------------
# Phase: inventory (plan A1-A3 + A5 pre-parity contingency)
# ---------------------------------------------------------------------------

_META_NAME_TOKENS = (
    "meta",
    "fingerprint",
    "summary",
    "manifest",
    "config",
    "report",
    "index",
    "readme",
)
_SIDECAR_STAGE_CAP_BYTES = 16 * 1024 * 1024


def _is_meta_sidecar(name: str) -> bool:
    low = name.lower()
    return low.endswith(".md") or any(tok in low for tok in _META_NAME_TOKENS)


def _stage_file(repo_path: str, min_free_note: str = "") -> Path:
    from explore_persona_space.orchestrate import hub

    target = STAGE_DIR / repo_path
    got = hub.stage_hub_file(REPO_DATA, repo_path, target, repo_type="dataset")
    _log(f"[stage] {repo_path} -> {got}{min_free_note}")
    return Path(got)


def _sidecar_payloads_by_dir(
    files: list[tuple[str, int]],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    """Stage + parse every .json meta-sidecar (size-capped); return payloads per dir.

    Returns (payloads[dirname][basename] -> parsed JSON, unparseable_paths).
    A malformed sidecar is RECORDED (never silently skipped) and classification
    for its directory degrades to path evidence.
    """
    payloads: dict[str, dict[str, object]] = {}
    unparseable: list[str] = []
    sidecars = [
        (path, size)
        for path, size in files
        if _is_meta_sidecar(Path(path).name) and Path(path).name.lower().endswith(".json")
    ]
    t0 = time.monotonic()
    for k, (path, size) in enumerate(sidecars, start=1):
        name = Path(path).name
        # Per-unit progress line (code-style intra-phase contract; round-1 CONCERN).
        _log(f"[stage] sidecar {k}/{len(sidecars)} {path} elapsed={time.monotonic() - t0:.1f}s")
        if size > _SIDECAR_STAGE_CAP_BYTES:
            unparseable.append(f"{path} (sidecar-too-large-not-staged: {size}B)")
            continue
        local = _stage_file(path)
        try:
            payload = json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            unparseable.append(f"{path} (sidecar-unparseable: {exc})")
            continue
        payloads.setdefault(str(Path(path).parent), {})[name] = payload
    return payloads, unparseable


_MODEL_KEYS = ("model", "model_instruct", "model_base", "model_name", "generating_model")

# Root-level model-family constants for the sibling-ladder roots (plan v4 § A0 evidence
# item "Sibling ladders ... different model families" + each producing issue's body).
# These trees are DIFFERENT model families, so none of their banks can be the
# Qwen2.5-7B base chat-template bank the A5 contingency looks for. NEVER derived from
# a bare `/base/` path token (round-1 blocker: #1336's generation/base/ is the LADDER
# RUNG named "base" on the Llama/Tulu family, not Qwen). Each constant is additionally
# verified at inventory time by a one-file row-schema probe per root
# (_probe_root_row_schema): any model-ish row field mentioning Qwen fails the phase.
ROOT_FAMILY = {
    "issue2061_sae_predictability": (
        "Tulu family (#2061 SAE ladder)",
        "plan v4 sibling-ladder evidence + #2061 body (Tulu SAE ladder)",
    ),
    "issue1902_stage_map": (
        "OLMo-2 family (#1902 stage ladder)",
        "plan v4 sibling-ladder evidence + #1902 body (OLMo-2 stage ladder)",
    ),
    "issue1336_rlvr_ladder": (
        "Llama/Tulu family (#1336 RLVR ladder)",
        "plan v4 sibling-ladder evidence + #1336 body (Llama/Tulu RLVR ladder; "
        "generation/<rung>/ segments base|sft|dpo|rlvr|rlvr_long are LADDER rungs, "
        "not Qwen model ids)",
    ),
}

# Known NON-completion path families, each excluded with an explicit reason
# (round-1 blocker: bare-extension classification labeled analysis tensors,
# allowlists, audits and corpora as completion banks).
_EXCLUDE_PATH_RULES = (
    (
        "/analysis_tensors/",
        "excluded:analysis-tensors (fit/eval tensors + row indexes, not completion banks)",
    ),
    (
        "/eval_results_mirror/",
        "excluded:eval-results-mirror (aggregated eval JSONs, not completion banks)",
    ),
    (
        "/corpus/",
        "excluded:corpus-input (prompt/seed corpus consumed by generation, not completions)",
    ),
    (
        "/steer_probe/",
        "excluded:steer-probe-artifact (steering-probe inputs/summaries, not generation banks)",
    ),
)
_FILTER_AUDIT_NAME_TOKENS = ("allowlist", "audit")


def _determine_from_sidecars(
    dir_payloads: dict[str, object],
) -> tuple[str | None, str | None, list[str]]:
    """Best-effort (model, render, evidence) from a directory's parsed sidecars."""
    model: str | None = None
    render: str | None = None
    evidence: list[str] = []
    for fname, payload in sorted(dir_payloads.items()):
        if not isinstance(payload, dict):
            continue
        for key in _MODEL_KEYS:
            val = payload.get(key)
            if isinstance(val, str) and val and model is None:
                model = val
                evidence.append(f"{fname}:{key}={val}")
        blob = json.dumps(payload, ensure_ascii=False)[:4000].lower()
        if render is None and ("chat_template" in blob or "apply_chat_template" in blob):
            render = "chat template"
            evidence.append(f"{fname}: chat-template token in sidecar")
        elif render is None and ("naturalistic" in blob or '"user: ' in blob):
            render = "plain User:/Assistant:"
            evidence.append(f"{fname}: plain-render token in sidecar")
    return model, render, evidence


def _classify_bank(path: str, dir_payloads: dict[str, object]) -> dict:
    """Classify one completion-bank file: model + render + provenance + evidence.

    Returns ``absence_decision`` in {"determinate", "family-excluded", "indeterminate"}:
    determinate = model AND render resolved from sidecars / producing-script evidence;
    family-excluded = a sibling-ladder root whose ROOT_FAMILY constant excludes Qwen
    (verified per root by the inventory row-schema probe); indeterminate = a plausible
    bank whose model or render could not be resolved — phase_inventory FAILS LOUD on
    any such row (the absence decision never defaults past an unresolved bank).
    """
    model: str | None = None
    render: str | None = None
    provenance = "on-policy generated"
    evidence: list[str] = []
    decision = "determinate"
    root = path.split("/", 1)[0]
    if "/track_s/" in path:
        meta = dir_payloads.get("track_s_meta.json")
        if isinstance(meta, dict):
            model = str(meta.get("model_instruct") or "")
            evidence.append(f"track_s_meta.json:model_instruct={model}")
            evidence.append(f"track_s_meta.json:sampling={json.dumps(meta.get('sampling'))}")
        render = "chat template"
        evidence.append(
            "issue825_gen_conversations._render: apply_chat_template(add_generation_prompt=True)"
        )
    elif f"{ARMG_ROOT}/" in path or "/turn_dynamics/" in path:
        if "/instruct/" in path:
            model = MODEL_INSTRUCT
        elif "/pretrained/" in path:
            model = MODEL_BASE
        render = "plain User:/Assistant: multi-turn transcript"
        evidence.append("path model token + armG rollout sidecars (plain-transcript rollout arm)")
    elif "/onpolicy_turn_depth/" in path:
        name = Path(path).name.lower()
        model = (
            MODEL_BASE if "pretrained" in name else (MODEL_INSTRUCT if "instruct" in name else None)
        )
        render = "plain User:/Assistant: (raw-text own-turn answers)"
        evidence.append("path token (#825 Result-6 own-answer arm; clarifier: raw-text render)")
    elif path.startswith("issue825_userbase_map/raw_completions/generation/"):
        meta = dir_payloads.get("conversations_meta.json")
        if isinstance(meta, dict) and meta.get("model_instruct"):
            model = str(meta["model_instruct"])
            evidence.append(f"conversations_meta.json:model_instruct={model}")
        render = "chat template"
        provenance = (
            "on-policy generated (instruct assistant turns; depth>=2 user turns "
            "simulated per conversations_meta.json u2_model)"
        )
        evidence.append(
            "producing script issue825_gen_conversations.py:202-203,522 "
            "(apply_chat_template(add_generation_prompt=True))"
        )
    elif root in ROOT_FAMILY:
        family, family_evidence = ROOT_FAMILY[root]
        model = f"{family} — NOT Qwen (root-family constant + row-schema probe)"
        evidence.append(family_evidence)
        if "__gen_naturalistic/" in path:
            render = "plain User:/Assistant: re-render (naturalistic twin dirs)"
        else:
            render = "family-native prompt render (non-Qwen family; not decision-relevant)"
        decision = "family-excluded"
    else:
        model, render, evidence = _determine_from_sidecars(dir_payloads)
    if decision == "determinate" and (model is None or render is None):
        decision = "indeterminate"
    is_base = bool(
        decision == "determinate"
        and model
        and "qwen2.5-7b" in model.lower()
        and "instruct" not in model.lower()
        and render == "chat template"
    )
    return {
        "classification": "completion-bank",
        "model": model or "undetermined",
        "render": render or "undetermined",
        "provenance": provenance,
        "evidence": evidence or ["no sidecar/path-family evidence"],
        "absence_decision": decision,
        "is_base_generated_chat_template_bank": is_base,
    }


def _classify_file(path: str, size: int, dir_payloads: dict[str, object]) -> dict:
    """Plan A3(a): classify ONE listed file (pure — no network; unit-tested).

    Pipeline order: non-text formats -> known non-completion path families (each
    with an explicit exclusion reason) -> filter/audit artifacts -> meta-sidecars
    -> the completion-bank classifier (_classify_bank).
    """
    name = Path(path).name
    suffix = Path(path).suffix.lower()
    row: dict = {"path": path, "size": size}
    if suffix not in (".json", ".jsonl", ".md"):
        row.update(
            classification=f"excluded:non-text-format ({suffix or 'no-extension'})",
            model=None,
            render=None,
        )
        return row
    slashed = f"/{path}"
    for token, label in _EXCLUDE_PATH_RULES:
        if token in slashed:
            row.update(classification=label, model=None, render=None)
            return row
    if any(tok in name.lower() for tok in _FILTER_AUDIT_NAME_TOKENS):
        row.update(
            classification=(
                "excluded:filter-audit-artifact "
                "(kept/dropped bookkeeping beside a bank, not completions)"
            ),
            model=None,
            render=None,
        )
        return row
    if _is_meta_sidecar(name):
        row.update(classification="meta-sidecar", model=None, render=None)
        return row
    row.update(_classify_bank(path, dir_payloads))
    return row


_ROOT_PROBE_CAP_BYTES = 32 * 1024 * 1024
_MODELISH_KEY_TOKENS = ("model", "engine", "checkpoint", "generator")


def _probe_root_row_schema(root: str, bank_files: list[tuple[str, int]]) -> dict:
    """Verify a ROOT_FAMILY constant against one real bank row (schema keys only).

    Stages the smallest bank file under the root (<=32MB; up to 3 candidates
    tried) and reads its first row: FAILS LOUD when any model-ish string field
    mentions Qwen (the family constant would then be contradicted). Records row-0
    KEYS + model-ish field values as manifest evidence — never row text (content
    hygiene: real-corpus completions are digest-only).
    """
    eligible = sorted((f for f in bank_files if f[1] <= _ROOT_PROBE_CAP_BYTES), key=lambda t: t[1])
    if not eligible:
        raise RuntimeError(
            f"inventory: root {root}: no bank file under the {_ROOT_PROBE_CAP_BYTES}B probe cap"
        )
    notes: list[str] = []
    for path, _size in eligible[:3]:
        local = _stage_file(path)
        try:
            if path.endswith(".jsonl"):
                rows = _read_jsonl(local)
                row0 = rows[0] if rows else None
            else:
                row0 = json.loads(local.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            notes.append(f"{path}: unparseable ({exc})")
            continue
        if not isinstance(row0, dict):
            notes.append(f"{path}: empty or non-dict row0")
            continue
        modelish = {
            k: v
            for k, v in row0.items()
            if isinstance(v, str) and any(t in k.lower() for t in _MODELISH_KEY_TOKENS)
        }
        for k, v in modelish.items():
            if "qwen" in v.lower():
                raise RuntimeError(
                    f"inventory: root {root} family constant CONTRADICTED — "
                    f"probe row field {k}={v!r} in {path}"
                )
        return {
            "probed_path": path,
            "row0_keys": sorted(str(k) for k in row0),
            "modelish_fields": modelish,
            "notes": notes,
        }
    raise RuntimeError(f"inventory: root {root}: no probe candidate yielded a row ({notes})")


def _format_table_rows(bank_rows: list[dict]) -> list[dict]:
    """Plan A3(b): one row per (task, base-row artifact), with committed R² evidence."""
    s2 = _read_committed_json("eval_results/issue_825/cells_S2.json")
    s1 = _read_committed_json("eval_results/issue_825/cells_S1.json")
    s2n = _read_committed_json("eval_results/issue_825/naturalistic-single-turn/cells_S2N.json")
    s1n = _read_committed_json("eval_results/issue_825/naturalistic-single-turn/cells_S1N.json")
    fc = _read_committed_json(
        "eval_results/issue_825/naturalistic-single-turn/format_contrast.json"
    )

    def _r2_19(payload: dict, label: str) -> float:
        arr = payload.get("r2_per_layer_obs")
        if not isinstance(arr, list) or len(arr) <= 19:
            raise KeyError(f"{label}: r2_per_layer_obs[19] missing (keys: {sorted(payload)[:20]})")
        return float(arr[19])

    fc_chat = fc["per_model"]["pretrained"]["paired_delta_frozen_layers"]["19"]["r2_obs_chat"]
    rows = [
        {
            "task": "#825 Result 1 (chat-template map)",
            "artifact": TRACK_S_JSONL,
            "generated_by_model": MODEL_INSTRUCT,
            "render": "chat template",
            "provenance": (
                "base row = pretrained model TEACHER-FORCED over instruct-generated "
                f"chat-rendered text (cells_S2.json cell={json.dumps(s2.get('cell'))})"
            ),
            "consuming_fit_result": (
                f"cells_S2.json r2_per_layer_obs[19]={_r2_19(s2, 'cells_S2')} (full n=5000); "
                f"cells_S1.json r2_per_layer_obs[19]={_r2_19(s1, 'cells_S1')}; restatement: "
                f"cells_S2N r2_per_layer_obs[19]={_r2_19(s2n, 'cells_S2N')}, "
                f"cells_S1N r2_per_layer_obs[19]={_r2_19(s1n, 'cells_S1N')}, "
                f"format_contrast pretrained chat@19={fc_chat} (n=4724 shared-subset refit)"
            ),
            "evidence_quote": "track_s_meta.json: model_instruct only, no base-model field",
        },
        {
            "task": "#825 Results 6-7 (turn-dynamics maps)",
            "artifact": f"{ARMG_ROOT}/pretrained/shard*/step*.jsonl",
            "generated_by_model": MODEL_BASE,
            "render": "plain User:/Assistant: multi-turn transcript",
            "provenance": "on-policy generated (base model wrote its own turns)",
            "consuming_fit_result": "#825 Results 6-7 turn-dynamics fits (raw plain render)",
            "evidence_quote": "armG answers open on the plain 'Assistant: ' header (plan pre-read)",
        },
        {
            "task": "#825 Results 6-7 (instruct comparator)",
            "artifact": f"{ARMG_ROOT}/instruct/shard*/step*.jsonl",
            "generated_by_model": MODEL_INSTRUCT,
            "render": "plain User:/Assistant: multi-turn transcript",
            "provenance": "on-policy generated",
            "consuming_fit_result": "#825 Results 6-7 turn-dynamics fits (instruct twin)",
            "evidence_quote": "same armG rollout arm, /instruct/ path token",
        },
        {
            "task": "#825 Result 6 (own-answer arm)",
            "artifact": "issue825_userbase_map/raw_completions/onpolicy_turn_depth/pretrained_own_turn_answers.jsonl",
            "generated_by_model": MODEL_BASE,
            "render": "plain User:/Assistant: (raw-text own-turn answers)",
            "provenance": "on-policy generated",
            "consuming_fit_result": "#825 Result 6 own-answer arm (inventory evidence only)",
            "evidence_quote": "filename token pretrained_own_turn_answers (clarifier: raw render)",
        },
    ]
    # #825 seed-conversations bank (raw_completions/generation) — one artifact, one row.
    conv_path = "issue825_userbase_map/raw_completions/generation/conversations.jsonl"
    conv = next((r for r in bank_rows if r["path"] == conv_path), None)
    if conv is not None:
        rows.append(
            {
                "task": "#825 seed conversations (track m)",
                "artifact": conv["path"],
                "generated_by_model": conv["model"],
                "render": conv["render"],
                "provenance": conv["provenance"],
                "consuming_fit_result": (
                    "seed conversations for the #825 turn-dynamics rollouts "
                    "(inventory evidence only)"
                ),
                "evidence_quote": "; ".join(conv["evidence"])[:400],
            }
        )
    # Collapsed rows: ONE row per (root, model-family, render) group — plan A3(b)
    # "one row per (task, base-row artifact)"; round-1 blocker: 561 per-FILE dynamic
    # rows violated that grain.
    static_prefixes = ("/track_s/", "/turn_dynamics/", "/onpolicy_turn_depth/")
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in bank_rows:
        if any(tok in row["path"] for tok in static_prefixes) or row["path"] == conv_path:
            continue
        root = next((r for r in ROOTS if row["path"].startswith(r)), row["path"].split("/")[0])
        grouped.setdefault((root, str(row["model"]), str(row["render"])), []).append(row)
    for (root, model, render), members in sorted(grouped.items()):
        prefix = os.path.commonpath([m["path"] for m in members])
        artifact = (
            members[0]["path"] if len(members) == 1 else f"{prefix}/** ({len(members)} files)"
        )
        rows.append(
            {
                "task": f"root {root}",
                "artifact": artifact,
                "generated_by_model": model,
                "render": render,
                "provenance": members[0]["provenance"],
                "consuming_fit_result": (
                    "inventory evidence only (collapsed per (task, artifact class); "
                    "classified from listing + root-family row-schema probe)"
                ),
                "evidence_quote": "; ".join(members[0]["evidence"])[:400],
            }
        )
    # Explicit zero-bank roots (e.g. a root holding only analysis tensors/encodings).
    for root in ROOTS:
        if not any(r["path"].startswith(root) for r in bank_rows):
            rows.append(
                {
                    "task": f"root {root}",
                    "artifact": f"{root}/** (0 completion banks)",
                    "generated_by_model": "N/A",
                    "render": "N/A",
                    "provenance": (
                        "no completion banks under this root "
                        "(analysis tensors / encodings / logs only)"
                    ),
                    "consuming_fit_result": "inventory evidence only",
                    "evidence_quote": (
                        "every file under this root classified excluded:* or meta-sidecar"
                    ),
                }
            )
    return rows


def _format_table_md(rows: list[dict]) -> str:
    cols = [
        "task",
        "artifact",
        "generated_by_model",
        "render",
        "provenance",
        "consuming_fit_result",
        "evidence_quote",
    ]
    lines = ["# #2477 format/provenance inventory (plan A3b)", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        vals = [str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def phase_inventory(args: argparse.Namespace) -> None:
    if args.smoke:
        raise SystemExit(
            "inventory has no smoke mode — it is a read-only listing phase; run it real"
        )
    outputs = [
        EVAL_DIR / "inventory_manifest.json",
        EVAL_DIR / "format_inventory.json",
        EVAL_DIR / "format_inventory.md",
    ]
    if not args.force and all(p.exists() for p in outputs):
        _log(
            "[phase=inventory] SKIP (idempotent): all three inventory artifacts exist — "
            "pass --force to re-run"
        )
        return
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    from explore_persona_space.orchestrate import hub

    _log("[phase=inventory] start")
    api = HfApi()
    all_files: list[tuple[str, int]] = []
    for root in ROOTS:
        entries = hub.retry_transient(
            lambda root=root: list(
                # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient at this call site
                api.list_repo_tree(
                    REPO_DATA, path_in_repo=root, repo_type="dataset", recursive=True
                )
            ),
            what=f"list_repo_tree({root})",
        )
        files = [(e.path, int(e.size or 0)) for e in entries if isinstance(e, RepoFile)]
        if not files:
            raise RuntimeError(
                f"inventory: root {root!r} listed ZERO files — refusing silent-empty"
            )
        _log(f"[inventory] root {root}: {len(files)} files")
        all_files.extend(files)

    payloads, unparseable = _sidecar_payloads_by_dir(all_files)

    manifest_rows: list[dict] = []
    bank_rows: list[dict] = []
    for path, size in sorted(all_files):
        row = _classify_file(path, size, payloads.get(str(Path(path).parent), {}))
        if row["classification"] == "completion-bank":
            bank_rows.append(row)
        manifest_rows.append(row)

    # FAIL LOUD before any artifact write: a plausible bank with an unresolved
    # model/render must block the absence decision, never default it (round-1 blocker).
    indeterminate = [r["path"] for r in bank_rows if r.get("absence_decision") == "indeterminate"]
    if indeterminate:
        raise RuntimeError(
            "inventory: ABSENCE DECISION BLOCKED — "
            f"{len(indeterminate)} plausible completion bank(s) with unresolved "
            f"model/render (first 20: {indeterminate[:20]})"
        )

    # Row-schema probe per family-excluded root: one real row's keys verify the
    # ROOT_FAMILY constant (raises loud if a model-ish field mentions Qwen).
    probes: dict[str, dict] = {}
    for root_dir in ROOT_FAMILY:
        fam_files = [
            (r["path"], int(r["size"]))
            for r in bank_rows
            if r["path"].startswith(root_dir + "/")
            and r.get("absence_decision") == "family-excluded"
        ]
        if fam_files:
            probes[root_dir] = _probe_root_row_schema(root_dir, fam_files)
            _log(
                f"[inventory] root-family probe {root_dir}: "
                f"keys={probes[root_dir]['row0_keys'][:12]} "
                f"modelish={probes[root_dir]['modelish_fields']}"
            )
        else:
            # Explicit vacuous record: every file under this root was excluded BEFORE the
            # bank classifier, so the family constant classified nothing — no row to probe.
            probes[root_dir] = {
                "probed_path": None,
                "note": "vacuous — zero family-excluded bank rows under this root",
            }
            _log(f"[inventory] root-family probe {root_dir}: vacuous (no bank rows)")

    candidates = [r["path"] for r in bank_rows if r.get("is_base_generated_chat_template_bank")]
    phase_c_fires_pre_parity = len(candidates) == 0
    n_by_class: dict[str, int] = {}
    n_by_exclusion_reason: dict[str, int] = {}
    for row in manifest_rows:
        cls = row["classification"]
        key = cls.split(" ")[0].split(":")[0]
        n_by_class[key] = n_by_class.get(key, 0) + 1
        if cls.startswith("excluded:"):
            reason = cls.split(" (")[0]
            n_by_exclusion_reason[reason] = n_by_exclusion_reason.get(reason, 0) + 1

    _write_json(
        EVAL_DIR / "inventory_manifest.json",
        {
            "metadata": _meta("inventory"),
            "roots": ROOTS,
            "n_files": len(manifest_rows),
            "n_by_class": n_by_class,
            "n_by_exclusion_reason": n_by_exclusion_reason,
            "unparseable_sidecars": unparseable,
            "root_family_probes": probes,
            "contingency": {
                "phase_c_fires_pre_parity": phase_c_fires_pre_parity,
                "candidate_base_chat_banks": candidates,
                "n_indeterminate_banks": 0,
                "note": (
                    "Phase C fires iff no file classifies as a Qwen2.5-7B base-GENERATED "
                    "chat-template completion bank; a candidate here still must pass the "
                    "A5 parity gate at --phase sample before Phase C is skipped. Any "
                    "plausible bank with unresolved model/render raises BEFORE this "
                    "manifest is written (absence never defaults)."
                ),
            },
            "files": manifest_rows,
        },
    )
    fmt_rows = _format_table_rows(bank_rows)
    _write_json(
        EVAL_DIR / "format_inventory.json",
        {"metadata": _meta("inventory"), "rows": fmt_rows},
    )
    (EVAL_DIR / "format_inventory.md").write_text(_format_table_md(fmt_rows), encoding="utf-8")
    _log(
        f"[phase=inventory] done: {len(manifest_rows)} files classified "
        f"({n_by_class}); base-chat bank candidates={candidates or 'NONE'}; "
        f"phase_c_fires_pre_parity={phase_c_fires_pre_parity}"
    )


# ---------------------------------------------------------------------------
# Phase: sample (plan A4 + A5 parity gate)
# ---------------------------------------------------------------------------


def _scan_track_s(path: Path) -> tuple[list[dict], dict]:
    """Full-corpus structural scan (plan assumption 9) — measured values returned."""
    rows = _read_jsonl(path)
    n_empty_prompt = sum(1 for r in rows if not str(r.get("prompt") or "").strip())
    n_empty_response = sum(1 for r in rows if not str(r.get("response") or "").strip())
    idxs = [r["prompt_idx"] for r in rows]
    stats = {
        "n_rows": len(rows),
        "n_empty_prompt": n_empty_prompt,
        "n_empty_response": n_empty_response,
        "prompt_idx_unique": len(set(idxs)) == len(idxs),
        "prompt_idx_range_0_4999": set(idxs) == set(range(5000)),
    }
    if len(rows) != 5000:
        raise RuntimeError(f"track_s scan: expected 5000 rows, got {len(rows)}")
    if n_empty_prompt or n_empty_response:
        raise RuntimeError(f"track_s scan: empty fields present: {stats}")
    if not stats["prompt_idx_unique"] or not stats["prompt_idx_range_0_4999"]:
        raise RuntimeError(f"track_s scan: prompt_idx not unique 0..4999: {stats}")
    return rows, stats


def _list_armg_step_files(api, model: str, shard: str) -> list[str]:
    from huggingface_hub.hf_api import RepoFile

    from explore_persona_space.orchestrate import hub

    root = f"{ARMG_ROOT}/{model}/{shard}"
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient at this call site
            api.list_repo_tree(REPO_DATA, path_in_repo=root, repo_type="dataset", recursive=True)
        ),
        what=f"list_repo_tree({root})",
    )
    files = sorted(
        e.path
        for e in entries
        if isinstance(e, RepoFile)
        and Path(e.path).name.startswith("step")
        and e.path.endswith(".jsonl")
    )
    if not files:
        raise RuntimeError(f"armG listing: no step*.jsonl under {root}")
    return files


def _load_armg_rows(api, model: str, shards: list[str]) -> dict[tuple[str, int], dict]:
    """(conv_id, depth) -> row for alive rows with non-empty user+answer (depth-1 excluded
    naturally by user=None). Duplicate keys fail loud (plan assumes uniqueness)."""
    out: dict[tuple[str, int], dict] = {}
    for shard in shards:
        step_files = _list_armg_step_files(api, model, shard)
        t0 = time.monotonic()
        for k, repo_path in enumerate(step_files, start=1):
            # Per-unit progress line (code-style intra-phase contract; round-1 CONCERN).
            _log(
                f"[sample] armG {model}/{shard} file {k}/{len(step_files)} "
                f"{Path(repo_path).name} elapsed={time.monotonic() - t0:.1f}s"
            )
            local = _stage_file(repo_path)
            for row in _read_jsonl(local):
                user = row.get("user")
                answer = row.get("answer")
                if row.get("alive") is not True:
                    continue
                if not isinstance(user, str) or not user.strip():
                    continue
                if not isinstance(answer, str) or not answer.strip():
                    continue
                key = (str(row["conv_id"]), int(row["depth"]))
                if key in out:
                    raise RuntimeError(
                        f"armG {model}: duplicate (conv_id, depth) key {key} "
                        f"(second source: {repo_path}) — plan A4 assumes unique keys"
                    )
                out[key] = row
    return out


def _merge_shard_rows(
    dst: dict[tuple[str, int], dict], src: dict[tuple[str, int], dict], model: str, shard: str
) -> None:
    """Merge a newly loaded shard's rows into the accumulated dict, failing loud on any
    cross-shard (conv_id, depth) key overlap (a bare dict.update would silently overwrite
    the earlier shard's row — plan A4 assumes unique keys across the whole armG store)."""
    overlap = set(dst) & set(src)
    if overlap:
        sample = sorted(overlap)[:5]
        raise RuntimeError(
            f"armG {model}: {len(overlap)} cross-shard duplicate (conv_id, depth) keys while "
            f"merging {shard} (e.g. {sample}) — plan A4 assumes unique keys"
        )
    dst.update(src)


def _bank_parity_gate(candidates: list[str], chat_items: list[dict]) -> dict:
    """Plan A5 bank-found parity gate: (i) prompt-panel identity, (ii) recipe parity,
    (iii) render parity. Returns the gate record; gate['skip_generation'] True only
    when a candidate passes ALL THREE. Any failure => Phase C fires anyway."""
    gate: dict = {"candidates": candidates, "skip_generation": False, "checks": []}
    if not candidates:
        return gate
    for cand in candidates:
        rec: dict = {"bank": cand, "passed": False, "failures": []}
        local = _stage_file(cand)
        bank_rows = _read_jsonl(local)
        by_idx = {r.get("prompt_idx"): r for r in bank_rows}
        # (i) prompt-panel identity
        missing = [it["prompt_idx"] for it in chat_items if it["prompt_idx"] not in by_idx]
        mismatched = [
            it["prompt_idx"]
            for it in chat_items
            if it["prompt_idx"] in by_idx
            and _sha(str(by_idx[it["prompt_idx"]].get("prompt") or "")) != _sha(it["prompt"])
        ]
        if missing or mismatched:
            rec["failures"].append(
                f"prompt-panel identity: {len(missing)} missing keys, {len(mismatched)} text mismatches"
            )
        # (ii) recipe parity — from the bank's sidecar(s) in the same dir
        sidecar_dir = str(Path(cand).parent)
        payloads, _ = _sidecar_payloads_by_dir([(cand, 0)])
        side = payloads.get(sidecar_dir, {})
        recipe_ok = False
        for payload in side.values():
            if not isinstance(payload, dict):
                continue
            sampling = payload.get("sampling") or {}
            model = str(payload.get("model") or payload.get("model_base") or "")
            if (
                sampling.get("temperature") == 1.0
                and sampling.get("top_p") == 0.95
                and sampling.get("max_tokens") == 1024
                and model == MODEL_BASE
            ):
                recipe_ok = True
        if not recipe_ok:
            rec["failures"].append(
                "recipe parity: no sidecar records the comparator recipe + base model"
            )
        # (iii) render parity — byte-for-byte hash vs the instruct-tokenizer chat render
        if not missing and not mismatched:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
            n_render_fail = 0
            for it in chat_items:
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": it["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                bank_rendered = by_idx[it["prompt_idx"]].get("rendered_prompt")
                if not isinstance(bank_rendered, str) or _sha(bank_rendered) != _sha(rendered):
                    n_render_fail += 1
            if n_render_fail:
                rec["failures"].append(
                    f"render parity: {n_render_fail}/200 rendered-prompt hash mismatches"
                )
        rec["passed"] = not rec["failures"]
        gate["checks"].append(rec)
        if rec["passed"]:
            gate["skip_generation"] = True
            gate["passing_bank"] = cand
            fresh = EVAL_DIR / "fresh_completions" / "base_chat_seed42.jsonl"
            keep = {it["prompt_idx"] for it in chat_items}
            _write_jsonl(fresh, [r for r in bank_rows if r.get("prompt_idx") in keep])
            break
    return gate


def phase_sample(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    smoke = bool(args.smoke)
    n_chat = 5 if smoke else N_CHAT
    n_pairs = 3 if smoke else N_RAW_PAIRS
    out_dir = Path("/tmp/issue-2477-smoke/samples") if smoke else (EVAL_DIR / "samples")
    _log(f"[phase=sample] start smoke={smoke} seed={args.seed} n_chat={n_chat} n_pairs={n_pairs}")

    # Idempotency guard (cheap phase; seeded => deterministic, but re-runs also re-upload the
    # manifest to HF — skip loud unless --force).
    if not smoke and not args.force and (out_dir / "sample_manifest.json").exists():
        _log(
            f"[phase=sample] SKIP (idempotent): {out_dir / 'sample_manifest.json'} exists — "
            "pass --force to re-run"
        )
        return

    inv_path = EVAL_DIR / "inventory_manifest.json"
    if not inv_path.exists():
        raise RuntimeError("sample: run --phase inventory first (inventory_manifest.json missing)")
    contingency = json.loads(inv_path.read_text(encoding="utf-8"))["contingency"]

    api = HfApi()
    track_local = _stage_file(TRACK_S_JSONL)
    _stage_file(TRACK_S_META)
    rows, scan_stats = _scan_track_s(track_local)
    _log(f"[sample] track_s full-corpus scan (assumption 9): {scan_stats}")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(5000, size=n_chat, replace=False)
    by_idx = {r["prompt_idx"]: r for r in rows}
    chat_items = [
        {
            "prompt_idx": int(i),
            "prompt": by_idx[int(i)]["prompt"],
            "instruct_response": by_idx[int(i)]["response"],
        }
        for i in idx
    ]

    shards = ["shard0of3"]
    pre = _load_armg_rows(api, "pretrained", shards)
    ins = _load_armg_rows(api, "instruct", shards)
    matched = sorted(set(pre) & set(ins))
    for extra in ("shard1of3", "shard2of3"):
        if len(matched) >= n_pairs:
            break
        _log(f"[sample] matched intersection {len(matched)} < {n_pairs}; extending to {extra}")
        shards.append(extra)
        _merge_shard_rows(pre, _load_armg_rows(api, "pretrained", [extra]), "pretrained", extra)
        _merge_shard_rows(ins, _load_armg_rows(api, "instruct", [extra]), "instruct", extra)
        matched = sorted(set(pre) & set(ins))
    realized_pairs = min(n_pairs, len(matched))
    if realized_pairs < n_pairs:
        _log(
            f"[sample] SHORTFALL: only {len(matched)} matched pairs available (take-all-and-report)"
        )
    pick = rng.choice(len(matched), size=realized_pairs, replace=False)
    pair_keys = [matched[int(i)] for i in pick]
    bad_ids = [cid for cid, _ in pair_keys if "__" in cid]
    if bad_ids:
        raise RuntimeError(
            f"sample: {len(bad_ids)} conv_ids contain '__' (judge item_id delimiter)"
        )

    def _pair_row(key: tuple[str, int]) -> dict:
        def _side(row: dict) -> dict:
            return {
                "user": row["user"],
                "answer": row["answer"],
                "finish_reason": row.get("finish_reason"),
                "n_gen_tokens": row.get("n_gen_tokens"),
            }

        return {
            "conv_id": key[0],
            "depth": key[1],
            "pretrained": _side(pre[key]),
            "instruct": _side(ins[key]),
        }

    parity_gate = {"candidates": [], "skip_generation": False, "checks": []}
    if not smoke and contingency["candidate_base_chat_banks"]:
        parity_gate = _bank_parity_gate(contingency["candidate_base_chat_banks"], chat_items)

    manifest = {
        "metadata": _meta("sample"),
        "meta": {
            "seed": int(args.seed),
            "smoke": smoke,
            "track_s_scan": scan_stats,
            "shards_used": shards,
            "n_chat_items": len(chat_items),
            "n_matched_pairs_available": len(matched),
            "realized_n_pairs": realized_pairs,
            "skip_generation": parity_gate["skip_generation"],
            "bank_parity_gate": parity_gate,
            "sampling_recipe": {
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": 1024,
                "seed": GEN_SEED,
            },
        },
        "chat_items": chat_items,
        "rawmt_pairs": [_pair_row(k) for k in pair_keys],
    }
    manifest_path = out_dir / "sample_manifest.json"
    _write_json(manifest_path, manifest)
    _log(f"[sample] wrote {manifest_path}")

    if smoke:
        _log("[phase=sample] done (smoke: scratch out-root, HF upload skipped by design)")
        return

    dest = f"{EXPERIMENT}/inputs/sample_manifest.json"
    url = hub._upload(
        manifest_path,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        dest,
        upload_as_file=True,
        raise_on_error=True,
    )
    if not url:
        raise RuntimeError(f"sample: manifest upload returned empty url for {dest}")
    missing = hub.verify_repo_paths_uploaded(
        api, hub.DEFAULT_DATASET_REPO, [dest], path_in_repo=f"{EXPERIMENT}/inputs"
    )
    if missing:
        raise RuntimeError(f"sample: manifest missing on Hub after upload: {missing}")
    _log(f"[phase=sample] done: manifest uploaded+verified at {hub.DEFAULT_DATASET_REPO}/{dest}")


# ---------------------------------------------------------------------------
# Phase: gen (plan C2-C4 — POD-SIDE ONLY; never run by the VM implementer)
# ---------------------------------------------------------------------------


def _vllm_generate_chunked(llm, prompts: list[str], sampling) -> list:
    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    outs = []
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        chunk = prompts[ci * chunk_size : (ci + 1) * chunk_size]
        _log(f"[vllm-chunk] gen chunk {ci + 1}/{n_chunks} ({len(chunk)} prompts)")
        outs.extend(llm.generate(chunk, sampling, use_tqdm=False))
    return outs


def _assert_gen_manifest_contract(manifest: dict, chat_items: list[dict], smoke: bool) -> None:
    """Fail loud BEFORE any tokenizer/LLM construction when the staged sample manifest does
    not carry the inputs gen is about to spend GPU time on (round-2 fix: the prior code
    validated nothing between manifest staging and LLM())."""
    errors: list[str] = []
    full = manifest.get("chat_items") or []
    if len(full) != N_CHAT:
        errors.append(f"manifest chat_items count: {len(full)} != expected {N_CHAT}")
    expected_n = 5 if smoke else N_CHAT
    if len(chat_items) != expected_n:
        errors.append(f"post-slice chat_items count: {len(chat_items)} != expected {expected_n}")
    non_int = sum(1 for it in chat_items if type(it.get("prompt_idx")) is not int)
    if non_int:
        errors.append(f"prompt_idx not a plain int on {non_int} items")
    idxs = [it.get("prompt_idx") for it in chat_items]
    if len(set(idxs)) != len(idxs):
        errors.append(f"prompt_idx not unique: {len(idxs) - len(set(idxs))} duplicates")
    n_empty = sum(
        1 for it in chat_items if not isinstance(it.get("prompt"), str) or not it["prompt"].strip()
    )
    if n_empty:
        errors.append(f"empty/non-str prompt on {n_empty} items")
    recipe = (manifest.get("meta") or {}).get("sampling_recipe") or {}
    # Manifest identity stays pinned to the PARENT recipe (temperature 1.0): the manifest
    # records how the sampled panel was drawn; the decoding-sensitivity arms deliberately
    # deviate at generation time via the registry, validated below (plan v5 §4 C0).
    expected_recipe = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 1024, "seed": GEN_SEED}
    for k, v in expected_recipe.items():
        if recipe.get(k) != v:
            errors.append(f"sampling_recipe[{k}]: {recipe.get(k)!r} != expected {v!r}")
    for set_name, rows in CONDITION_SETS.items():
        for slug, _render, temp, _stop in rows:
            if type(temp) is not float or not (0.0 <= temp <= 2.0):
                errors.append(
                    f"CONDITION_SETS[{set_name}][{slug}]: temperature {temp!r} "
                    "not a float in [0.0, 2.0]"
                )
    if errors:
        raise RuntimeError(
            "gen manifest contract failed (" + str(len(errors)) + " errors): " + "; ".join(errors)
        )


def phase_gen(args: argparse.Namespace) -> None:
    smoke = bool(args.smoke)
    out_root = Path(args.out)
    if args.condition_set == DECSENS and not smoke and out_root.name != "decoding_sensitivity":
        # Fail fast: a shared out root would cross-fire the PARENT's gen_done.json idempotency
        # guard (silent SKIP of a paid phase) and collide nothing else loudly (plan v5 §4:
        # the decoding-sensitivity sentinel is path-distinct BY out-root).
        raise SystemExit(
            "gen --condition-set decoding-sensitivity requires the set-specific --out "
            "(.../decoding_sensitivity, plan v5 phase_outputs): a shared out root would "
            "cross-fire the parent's gen_done.json idempotency guard"
        )
    # Idempotency guard: gen is the paid GPU phase. Placed BEFORE the torch import so the
    # skip leg is VM-smokable (refusing a duplicate paid run needs no CUDA); emits the
    # [phase=done] breadcrumb so a poller treats the skip as a completed phase.
    if not smoke and not args.force and (out_root / "gen_done.json").exists():
        _log(
            f"[phase=gen] SKIP (idempotent): {out_root / 'gen_done.json'} exists — "
            "pass --force to re-run"
        )
        _log("[phase=done]")
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("gen is pod-side (GPU) only — refuse to run without CUDA")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.orchestrate import hub

    out_root.mkdir(parents=True, exist_ok=True)
    _log(f"[phase=gen] start smoke={smoke} out={out_root}")

    manifest_path = out_root / "sample_manifest.json"
    hub.stage_hub_file(
        REPO_DATA, f"{EXPERIMENT}/inputs/sample_manifest.json", manifest_path, repo_type="dataset"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["meta"]["skip_generation"]:
        raise RuntimeError(
            "gen: sample_manifest records skip_generation=true (A5 bank parity passed) — "
            "Phase C must not run; the banked arm substitutes the fresh file"
        )
    chat_items = manifest["chat_items"]
    if smoke:
        chat_items = chat_items[:5]
    # Round-2 fix 4: validate the manifest contract BEFORE tokenizers/LLM spend anything.
    _assert_gen_manifest_contract(manifest, chat_items, smoke)

    tok_i = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
    tok_b = AutoTokenizer.from_pretrained(MODEL_BASE)
    probe = tok_i.apply_chat_template(
        [{"role": "user", "content": "smoke render parity probe"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if tok_b.encode(probe) != tok_i.encode(probe):
        raise RuntimeError(
            "gen: base/instruct tokenizers disagree on the chat render (parity assert)"
        )
    im_end_b = tok_b.encode("<|im_end|>", add_special_tokens=False)
    im_end_i = tok_i.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end_b) != 1 or im_end_b != im_end_i:
        raise RuntimeError(f"gen: <|im_end|> not a single shared token ({im_end_b} vs {im_end_i})")

    prompts_raw = [it["prompt"] for it in chat_items]
    rendered_chat = [
        tok_i.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts_raw
    ]
    # No trailing space after "Assistant:" — the trailing-space token BPE-merges into the
    # first content token (issue825_render_formats.py:234-236; issue1336_render.py:168).
    rendered_bare = [f"User: {p}\n\nAssistant:" for p in prompts_raw]

    llm = LLM(model=MODEL_BASE)  # engine defaults: issue825_gen_conversations._build_llm parity
    meta_common = _meta("gen")
    # C0: arms come from the condition-set registry. "parent" reproduces the parent round's
    # two literals at temperature 1.0; "decoding-sensitivity" loops its 4 conditions in this
    # ONE engine session (plan v5 §4: one LLM() load, one batched generate per condition).
    cset = CONDITION_SETS[args.condition_set]
    render_map = {"chat": rendered_chat, "bare": rendered_bare}
    if args.condition_set == DECSENS:
        _log(f"[gen] condition_set={args.condition_set}: {len(cset)} conditions")
    gen_stats: dict[str, dict] = {}
    suffix = "smoke" if smoke else "seed42"
    for arm, render_key, temp, stop_t in cset:
        rendered = render_map[render_key]
        stop = list(stop_t)
        sampling = SamplingParams(
            n=1, temperature=temp, top_p=0.95, max_tokens=1024, seed=GEN_SEED, stop=stop
        )
        outs = _vllm_generate_chunked(llm, rendered, sampling)
        rows = []
        for it, rprompt, o in zip(chat_items, rendered, outs, strict=True):
            comp = o.outputs[0]
            rows.append(
                {
                    "prompt_idx": it["prompt_idx"],
                    "prompt": it["prompt"],
                    "rendered_prompt": rprompt,
                    "response": comp.text,
                    "finish_reason": comp.finish_reason,
                    "n_gen_tokens": len(comp.token_ids),
                    "arm": arm,
                    "model": MODEL_BASE,
                    "sampling": {
                        "n": 1,
                        "temperature": temp,
                        "top_p": 0.95,
                        "max_tokens": 1024,
                        "seed": GEN_SEED,
                        "stop": stop,
                    },
                    "git_commit": meta_common["git_commit"],
                }
            )
        n_empty = sum(1 for r in rows if not r["response"].strip())
        cap_hit = sum(1 for r in rows if r["finish_reason"] == "length") / len(rows)
        gen_stats[arm] = {
            "n": len(rows),
            "n_empty": n_empty,
            "cap_hit_fraction": cap_hit,
            "finish_reasons": {
                # key=str: finish_reason can be None (vLLM in-flight abort) — a bare sorted()
                # over a mixed {None, str} set raises TypeError (round-2 opportunistic fix).
                str(fr): sum(1 for r in rows if r["finish_reason"] == fr)
                for fr in sorted({r["finish_reason"] for r in rows}, key=str)
            },
        }
        if smoke and n_empty:
            raise RuntimeError(
                f"gen smoke: {n_empty}/5 EMPTY responses in {arm} (G1 non-empty assert)"
            )
        # Checkpoint per arm: JSONL written the moment the arm completes (plan C3).
        path = out_root / ("smoke" if smoke else "") / f"{arm}_{suffix}.jsonl"
        _write_jsonl(path, rows)
        _log(f"[gen] arm {arm}: {gen_stats[arm]}")

    gen_meta = {
        "metadata": meta_common,
        "smoke": smoke,
        "model": MODEL_BASE,
        "render_parity_assert": "token-id equality on smoke render; <|im_end|> single shared token",
        "arms": gen_stats,
        "stop_strings": {a: list(s) for a, _r, _t, s in cset},
    }
    if args.condition_set == DECSENS:
        gen_meta["condition_set"] = args.condition_set
        gen_meta["realized_temperatures"] = {a: t for a, _r, t, _s in cset}
        gen_meta["inert_fields_note"] = (
            "temperature 0.0 arms decode greedily under vLLM; top_p=0.95 and seed=42 are "
            "inert there (recorded for recipe parity, not load-bearing)"
        )
    if smoke:
        _write_json(out_root / "smoke" / "gen_meta_smoke.json", gen_meta)
        _log("[phase=gen] done (smoke — no upload; eyeball smoke/*.jsonl for stop behavior)")
        return
    _write_json(out_root / "gen_meta.json", gen_meta)

    # C4: upload BEFORE anything else. Explicit per-file hub._upload calls (the parent's
    # unrolled shape generalized to the condition set's file list — same calls, same order,
    # same kwargs on the parent set) because the canonical upload_raw_completions_to_data_repo
    # helper composes dests as <exp>/raw_completions/<rel> with selection requiring a local
    # raw_completions/ dir — which would double the prefix vs the plan-declared destination.
    dest_prefix = (
        DECSENS_HUB_GEN_PREFIX
        if args.condition_set == DECSENS
        else f"{EXPERIMENT}/raw_completions/generation"
    )
    jsonl_paths = [out_root / f"{arm}_{suffix}.jsonl" for arm, _r, _t, _s in cset]
    meta_path = out_root / "gen_meta.json"
    for p in [*jsonl_paths, meta_path]:
        url = hub._upload(
            p,
            hub.DEFAULT_DATASET_REPO,
            "dataset",
            f"{dest_prefix}/{p.name}",
            upload_as_file=True,
            raise_on_error=True,
        )
        if not url:
            raise RuntimeError(f"gen: upload of {p.name} returned an empty url")
    from huggingface_hub import HfApi

    expected = [f"{dest_prefix}/{p.name}" for p in [*jsonl_paths, meta_path]]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), hub.DEFAULT_DATASET_REPO, expected, path_in_repo=dest_prefix
    )
    if missing:
        raise RuntimeError(f"gen: files missing on Hub after upload: {missing}")

    # Mirror the small text files onto the issue branch tree (plan C4 / v5 §4).
    mirror = (
        DECSENS_EVAL_DIR / "fresh_completions"
        if args.condition_set == DECSENS
        else REPO_ROOT / "eval_results" / "issue_2477" / "fresh_completions"
    )
    mirror.mkdir(parents=True, exist_ok=True)
    for p in jsonl_paths:
        shutil.copy2(p, mirror / p.name)
    _write_json(
        out_root / "gen_done.json",
        {"metadata": meta_common, "uploaded": expected, "arms": gen_stats, "status": "done"},
    )
    _log("[phase=gen] done: uploaded+verified; sentinel gen_done.json written")
    _log("[phase=done]")


# ---------------------------------------------------------------------------
# Judge arms (shared by pilot / wave / aggregate)
# ---------------------------------------------------------------------------


@dataclass
class ArmData:
    name: str
    items: list[tuple[str, str, str]] = field(default_factory=list)
    pair_key: dict[str, object] = field(default_factory=dict)  # item_id -> pairing key
    depth: dict[str, int] = field(default_factory=dict)  # raw arms only
    cap_hit: dict[str, bool] = field(default_factory=dict)  # where finish_reason recorded
    cap_hit_note: str | None = None
    strip_counts: dict[str, int] = field(default_factory=dict)


def _load_manifest() -> dict:
    path = EVAL_DIR / "samples" / "sample_manifest.json"
    if not path.exists():
        raise RuntimeError("judge: run --phase sample first (sample_manifest.json missing)")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_fresh_rows_contract(rows: list[dict], manifest: dict, name: str) -> None:
    """Verify a fresh-completions file (pod-local mirror OR HF-staged) matches the sample
    manifest's panel before any judge spend (round-2 fix 1)."""
    chat_items = manifest["chat_items"]
    if len(rows) != len(chat_items):
        raise RuntimeError(
            f"judge: {name} row count {len(rows)} != manifest chat_items {len(chat_items)}"
        )
    want = {it["prompt_idx"] for it in chat_items}
    got = {r.get("prompt_idx") for r in rows}
    if got != want:
        missing = sorted(want - got, key=str)[:5]
        extra = sorted(got - want, key=str)[:5]
        raise RuntimeError(
            f"judge: {name} prompt_idx set mismatch vs manifest "
            f"(missing e.g. {missing}, extra e.g. {extra})"
        )


def _fresh_rows(name: str, manifest: dict, condition_set: str = "parent") -> list[dict]:
    fresh_dir = (
        DECSENS_EVAL_DIR / "fresh_completions"
        if condition_set == DECSENS
        else EVAL_DIR / "fresh_completions"
    )
    path = fresh_dir / name
    if not path.exists():
        # Round-2 fix 1: judge phases may run on a different machine than gen — the pod-local
        # mirror is the fast path; the plan-declared HF dest is the permanent source
        # (plan §off_pod_phases). A file absent on the Hub too means Phase C never ran.
        from huggingface_hub.utils import EntryNotFoundError

        from explore_persona_space.orchestrate import hub

        repo_prefix = (
            DECSENS_HUB_GEN_PREFIX
            if condition_set == DECSENS
            else f"{EXPERIMENT}/raw_completions/generation"
        )
        repo_path = f"{repo_prefix}/{name}"
        _log(f"[judge] {path} absent locally — staging from {REPO_DATA}/{repo_path}")
        try:
            hub.stage_hub_file(REPO_DATA, repo_path, path, repo_type="dataset")
        except EntryNotFoundError as exc:
            raise RuntimeError(
                f"judge: {name} absent locally AND on the Hub at {repo_path} — "
                "Phase C (gen) has not produced fresh completions yet"
            ) from exc
    rows = _read_jsonl(path)
    _validate_fresh_rows_contract(rows, manifest, name)
    return rows


def _fresh_arm(
    arm_name: str, fname: str, manifest: dict, prompt_by_idx: dict, condition_set: str
) -> ArmData:
    """One fresh-completions arm: panel/prompt-hash asserts + cap-hit capture (plan B1).

    Factored verbatim from the parent build_arms fresh loop (same asserts, same messages);
    condition_set threads only the _fresh_rows local-dir/Hub-prefix routing.
    """
    chat_items = manifest["chat_items"]
    a = ArmData(name=arm_name)
    for row in _fresh_rows(fname, manifest, condition_set=condition_set):
        p_idx = row["prompt_idx"]
        if p_idx not in prompt_by_idx:
            raise RuntimeError(f"{arm_name}: fresh row prompt_idx={p_idx} not in the sampled panel")
        if _sha(row["prompt"]) != _sha(prompt_by_idx[p_idx]):
            raise RuntimeError(
                f"{arm_name}: prompt text mismatch vs manifest at prompt_idx={p_idx}"
            )
        iid = f"{arm_name}--{p_idx}"
        a.items.append((iid, row["prompt"], row["response"]))
        a.pair_key[iid] = p_idx
        fr = row.get("finish_reason")
        if fr is not None:
            a.cap_hit[iid] = fr == "length"
    if len(a.items) != len(chat_items):
        raise RuntimeError(
            f"{arm_name}: {len(a.items)} fresh rows vs {len(chat_items)} sampled prompts"
        )
    return a


def build_arms(manifest: dict, condition_set: str = "parent") -> dict[str, ArmData]:
    """Assemble the judged arms; item_id grammar per plan B1 ('--' delimiter).

    condition_set="parent" (default): the five §5 arms (banked instruct + fresh base pair +
    raw multi-turn). condition_set="decoding-sensitivity": the four fresh temperature-ablation
    arms (plan v5 §4) — panel + prompt-hash asserts unchanged, no banked arms.
    """
    arms: dict[str, ArmData] = {}

    chat_items = manifest["chat_items"]
    prompt_by_idx = {it["prompt_idx"]: it["prompt"] for it in chat_items}

    if condition_set == DECSENS:
        for slug, _render, _temp, _stop in CONDITION_SETS[DECSENS]:
            arm_name = f"arm_{slug}"
            arms[arm_name] = _fresh_arm(
                arm_name, f"{slug}_seed42.jsonl", manifest, prompt_by_idx, DECSENS
            )
        for name in DECSENS_ARM_NAMES:
            if not arms[name].items:
                raise RuntimeError(f"build_arms: arm {name} is EMPTY — refusing silent-empty arm")
        return arms

    a = ArmData(name="arm_instruct_chat", cap_hit_note="N/A — not recorded in the banked artifact")
    for it in chat_items:
        iid = f"arm_instruct_chat--{it['prompt_idx']}"
        a.items.append((iid, it["prompt"], it["instruct_response"]))
        a.pair_key[iid] = it["prompt_idx"]
    arms[a.name] = a

    for arm_name, fname in (
        ("arm_base_chat", "base_chat_seed42.jsonl"),
        ("arm_base_bare", "base_bare_seed42.jsonl"),
    ):
        arms[arm_name] = _fresh_arm(arm_name, fname, manifest, prompt_by_idx, "parent")

    for arm_name, side in (("arm_base_rawmt", "pretrained"), ("arm_instruct_rawmt", "instruct")):
        a = ArmData(name=arm_name)
        n_strip_user = n_strip_ans = 0
        for pair in manifest["rawmt_pairs"]:
            row = pair[side]
            question, s_u = _strip_header(row["user"], "User: ")
            answer, s_a = _strip_header(row["answer"], "Assistant: ")
            n_strip_user += int(s_u)
            n_strip_ans += int(s_a)
            iid = f"{arm_name}--{pair['conv_id']}--d{pair['depth']}"
            if "__" in iid:
                raise RuntimeError(f"{arm_name}: item_id contains '__': {iid!r}")
            a.items.append((iid, question, answer))
            a.pair_key[iid] = (pair["conv_id"], pair["depth"])
            a.depth[iid] = int(pair["depth"])
            fr = row.get("finish_reason")
            if fr is not None:
                a.cap_hit[iid] = fr == "length"
        a.strip_counts = {
            "user_header_stripped": n_strip_user,
            "answer_header_stripped": n_strip_ans,
        }
        arms[arm_name] = a

    for name in ARM_NAMES:
        if not arms[name].items:
            raise RuntimeError(f"build_arms: arm {name} is EMPTY — refusing silent-empty arm")
    return arms


# ---------------------------------------------------------------------------
# Phase: judge-pilot (plan B4)
# ---------------------------------------------------------------------------


def phase_judge_pilot(args: argparse.Namespace) -> None:
    if args.smoke:
        raise SystemExit("judge-pilot has no smoke mode — the pilot IS the tiny gated pre-wave")
    decsens = args.condition_set == DECSENS
    judge_dir = (DECSENS_EVAL_DIR / "judge") if decsens else (EVAL_DIR / "judge")
    # Idempotency guard (paid API phase): skip ONLY on a PASSED pilot — a failed pilot
    # re-runs by design (that IS the fix path after a rubric/instrument change).
    pilot_path = judge_dir / "pilot_report.json"
    if not args.force and pilot_path.exists():
        prior = json.loads(pilot_path.read_text(encoding="utf-8"))
        if prior.get("passed"):
            _log(
                f"[phase=judge-pilot] SKIP (idempotent): {pilot_path} exists with passed=true — "
                "pass --force to re-run"
            )
            return
        _log(
            f"[phase=judge-pilot] prior pilot_report has passed={prior.get('passed')} — re-running"
        )
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    _log("[phase=judge-pilot] start")
    manifest = _load_manifest()
    arms = build_arms(manifest, condition_set=args.condition_set)
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    cache_slug = "judge_cache_decsens_pilot" if decsens else "judge_cache_pilot"
    report = judge_pilot_gate(
        {name: a.items for name, a in arms.items()},
        RUBRIC,
        max_tokens=JUDGE_MAX_TOKENS,
        cache_dir=REPO_ROOT / "data" / "issue_2477" / cache_slug / run_ts,
        save_raw_dir=judge_dir / "pilot_raw",
        n_draws=N_DRAWS_PILOT,
        target_total_draws=(
            DECSENS_PILOT_TARGET_TOTAL_DRAWS if decsens else PILOT_TARGET_TOTAL_DRAWS
        ),
        parse_fail_threshold=0.02,
        min_effective_draws_per_arm=10,
        wave_threshold_base=0,
        report_path=judge_dir / "pilot_report.json",
        seed=0,
    )
    _log(f"[judge-pilot] verdict={report.verdict} passed={report.passed}")
    for line in report.failures:
        _log(f"[judge-pilot] FAILURE: {line}")
    for line in report.warnings:
        _log(f"[judge-pilot] warning: {line}")
    if not report.passed:
        raise SystemExit(7)  # designed gate refusal — distinct rc, report JSON already written
    _log("[phase=judge-pilot] done: PASS")


# ---------------------------------------------------------------------------
# Phase: judge-wave (plan B3 smoke / B5 production)
# ---------------------------------------------------------------------------


def _judge_wave_complete(judge_dir: Path, expected: dict[str, set[str]]) -> bool:
    """True only when EVERY arm's judge_raw file exists, parses, and its all_scores keys
    (``{item_id}__{idx:05d}__{draw:02d}``, batch_judge custom_id grammar) cover exactly the
    expected item-id set — the round-2 idempotency predicate for the paid wave."""
    for name, want in expected.items():
        path = judge_dir / f"judge_raw_{name}.json"
        if not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return False
        got = {k.rsplit("__", 2)[0] for k in (raw.get("all_scores") or {})}
        if got != want:
            return False
    return True


def phase_judge_wave(args: argparse.Namespace) -> None:
    from explore_persona_space.eval.graded_judge import judge_graded

    from explore_persona_space.orchestrate import hub

    manifest = _load_manifest()
    run_ts = time.strftime("%Y%m%d-%H%M%S")

    if args.smoke:
        # B3/B3': live forced-batch smoke through the run's exact request builder (the probe
        # validates the request/parse contract, not arm content — items are shared).
        _log("[phase=judge-wave] B3 live forced-batch smoke (5 items x 1 draw, threshold_base=0)")
        scratch = Path("/tmp/issue-2477-smoke") / (
            "judge_decsens" if args.condition_set == DECSENS else "judge"
        )
        items = [
            (f"smoke--{it['prompt_idx']}", it["prompt"], it["instruct_response"])
            for it in manifest["chat_items"][:5]
        ]
        save_raw = scratch / "judge_raw_smoke.json"
        result = judge_graded(
            items,
            RUBRIC,
            n_draws=1,
            cache_dir=scratch / "cache" / run_ts,
            save_raw=save_raw,
            max_tokens=JUDGE_MAX_TOKENS,
            threshold_base=0,
        )
        raw_text = save_raw.read_text(encoding="utf-8")
        if "invalid_request_error" in raw_text:
            raise RuntimeError(
                "B3 smoke: save_raw contains invalid_request_error — request shape bad"
            )
        n_scored = sum(1 for s in result.scores.values() if s is not None)
        if n_scored < 4:
            raise RuntimeError(
                f"B3 smoke: only {n_scored}/5 items scored "
                f"(dropped={result.n_dropped_draws}, transport={result.n_transport_lost_draws}, "
                f"api_refusal={result.n_api_refusal_draws})"
            )
        _log(
            f"[phase=judge-wave] B3 smoke PASS: {n_scored}/5 scored, "
            f"stop_reason_tally={result.stop_reason_tally}"
        )
        return

    decsens = args.condition_set == DECSENS
    judge_dir = (DECSENS_EVAL_DIR / "judge") if decsens else (EVAL_DIR / "judge")
    cache_slug = "judge_cache_decsens" if decsens else "judge_cache"
    arms = build_arms(manifest, condition_set=args.condition_set)

    # Idempotency guard (paid API phase): skip ONLY when the COMPLETE per-set output arm set
    # exists and each file covers exactly this manifest's item ids — a partial wave
    # (crash between arms) re-runs (round-2 fix 3).
    expected_ids = {name: {iid for iid, _q, _a in arm.items} for name, arm in arms.items()}
    if not args.force and _judge_wave_complete(judge_dir, expected_ids):
        _log(
            f"[phase=judge-wave] SKIP (idempotent): all {len(expected_ids)} arms' judge_raw "
            "files complete for this manifest — pass --force to re-run"
        )
        return

    pilot_path = judge_dir / "pilot_report.json"
    if not pilot_path.exists():
        raise RuntimeError("judge-wave: run --phase judge-pilot first (pilot_report.json missing)")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not pilot.get("passed"):
        raise RuntimeError("judge-wave: pilot gate did not PASS — fix + re-pilot before the wave")

    for name, arm in arms.items():
        save_raw = judge_dir / f"judge_raw_{name}.json"
        _log(
            f"[judge-wave] arm {name}: {len(arm.items)} items x {N_DRAWS_PROD} draws (batch-pinned)"
        )
        result = judge_graded(
            arm.items,
            RUBRIC,
            n_draws=N_DRAWS_PROD,
            cache_dir=REPO_ROOT / "data" / "issue_2477" / cache_slug / run_ts / name,
            save_raw=save_raw,
            max_tokens=JUDGE_MAX_TOKENS,
            threshold_base=0,
        )
        _log(
            f"[judge-wave] arm {name} done: total={result.n_total_draws} "
            f"content_drops={result.n_dropped_draws} transport={result.n_transport_lost_draws} "
            f"api_refusal={result.n_api_refusal_draws} "
            f"frac_items_complete={result.frac_items_complete:.4f}"
        )

    url = hub._upload(
        judge_dir,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        DECSENS_HUB_JUDGE_PREFIX if decsens else f"{EXPERIMENT}/judge_raw",
        raise_on_error=True,
    )
    if not url:
        raise RuntimeError("judge-wave: judge_raw folder upload returned empty url")
    _log(f"[phase=judge-wave] done: raw draws mirrored to {url}")


# ---------------------------------------------------------------------------
# Phase: aggregate (plan B6)
# ---------------------------------------------------------------------------


def _arm_stats(arm: ArmData, save_raw: Path) -> dict:
    from scipy.stats import spearmanr

    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    jr = judge_result_from_save_raw(save_raw, arm.items)
    kept = {iid: s for iid, s in jr.scores.items() if s is not None}
    answers = {iid: ans for iid, _q, ans in arm.items}
    kept_ids = sorted(kept)
    scores = [kept[i] for i in kept_ids]
    d3 = [_distinct_3gram_rate([answers[i]]) for i in kept_ids]
    if len(kept_ids) >= 3 and len(set(scores)) > 1 and len(set(d3)) > 1:
        rho, pval = spearmanr(scores, d3)
        spear = {"rho": float(rho), "p": float(pval), "n": len(kept_ids)}
    else:
        spear = {"rho": None, "p": None, "n": len(kept_ids), "note": "degenerate/constant inputs"}

    n_items = len(arm.items)
    n_kept = len(kept_ids)
    n_coherent = sum(1 for s in scores if s >= COHERENT_THRESHOLD)
    frac = (n_coherent / n_kept) if n_kept else float("nan")
    frac_sens = n_coherent / n_items  # drops counted as incoherent
    per_depth: dict[str, float] = {}
    if arm.depth:
        for d in sorted(set(arm.depth.values())):
            vals = [kept[i] for i in kept_ids if arm.depth.get(i) == d]
            if vals:
                per_depth[str(d)] = float(np.mean(vals))
    if arm.cap_hit:
        cap_hit = sum(arm.cap_hit.values()) / len(arm.cap_hit)
        cap_note = f"over {len(arm.cap_hit)}/{n_items} items with finish_reason recorded"
    else:
        cap_hit = None
        cap_note = arm.cap_hit_note or "N/A — finish_reason not recorded"
    return {
        "n_items": n_items,
        "n_items_kept": n_kept,
        "n_items_zero_valid": n_items - n_kept,
        "frac_items_complete": jr.frac_items_complete,
        "n_total_draws": jr.n_total_draws,
        "drop_counts": {
            "content": jr.n_dropped_draws,
            "content_refusal_subset": jr.n_refusal_draws,
            "content_truncation_subset": jr.n_truncation_dropped_draws,
            "transport": jr.n_transport_lost_draws,
            "api_refusal": jr.n_api_refusal_draws,
        },
        "stop_reason_tally": jr.stop_reason_tally,
        "mean": float(np.mean(scores)) if scores else None,
        "mean_ci95": _boot_ci_mean(scores),
        "frac_coherent": {
            "value": frac,
            "wilson_ci95": _wilson_ci(n_coherent, n_kept),
            "n_coherent": n_coherent,
            "n_kept": n_kept,
        },
        "frac_coherent_drops_as_incoherent": {
            "value": frac_sens,
            "wilson_ci95": _wilson_ci(n_coherent, n_items),
        },
        "distinct_3gram": {
            "mean": float(np.mean(d3)) if d3 else None,
            "spearman_vs_score": spear,
        },
        "cap_hit_fraction": cap_hit if cap_hit is not None else cap_note,
        "per_depth_mean": per_depth or None,
        "strip_counts": arm.strip_counts or None,
        "kept_scores": {i: kept[i] for i in kept_ids},
        "kept_d3": {i: float(v) for i, v in zip(kept_ids, d3)},
    }


def _paired_delta(a: dict, b: dict, arm_a: ArmData, arm_b: ArmData) -> dict:
    """Per-key delta a-b over pairs kept on BOTH sides (plan B6 excluded-pair counts)."""
    ka = {arm_a.pair_key[i]: s for i, s in a["kept_scores"].items()}
    kb = {arm_b.pair_key[i]: s for i, s in b["kept_scores"].items()}
    # JSON round-trips tuple keys to lists — normalize to tuples for the join.
    ka = {tuple(k) if isinstance(k, list) else k: v for k, v in ka.items()}
    kb = {tuple(k) if isinstance(k, list) else k: v for k, v in kb.items()}
    shared = sorted(set(ka) & set(kb), key=str)
    deltas = [ka[k] - kb[k] for k in shared]
    n_panel = max(a["n_items"], b["n_items"])
    return {
        "n_pairs": len(deltas),
        "n_excluded_pairs": n_panel - len(deltas),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "delta_ci95": _boot_ci_mean(deltas),
        # Per-pair deltas persisted so --phase figures is a pure consumer of the
        # verdict JSON (paired-delta histogram, plan B7/§6 exploratory dump).
        "per_pair_delta": {str(k): float(ka[k] - kb[k]) for k in shared},
    }


def _aggregate_core(
    arms: dict[str, ArmData], save_raw_paths: dict[str, Path], out_json: Path
) -> dict:
    per_arm = {name: _arm_stats(arm, save_raw_paths[name]) for name, arm in arms.items()}
    if "arm_base_chat" not in per_arm:
        raise RuntimeError("aggregate: arm_base_chat missing — the verdict arm is required")

    pairs = {}
    if "arm_instruct_chat" in per_arm:
        pairs["base_chat_minus_instruct_chat"] = _paired_delta(
            per_arm["arm_base_chat"],
            per_arm["arm_instruct_chat"],
            arms["arm_base_chat"],
            arms["arm_instruct_chat"],
        )
    if "arm_base_rawmt" in per_arm and "arm_instruct_rawmt" in per_arm:
        pairs["base_rawmt_minus_instruct_rawmt"] = _paired_delta(
            per_arm["arm_base_rawmt"],
            per_arm["arm_instruct_rawmt"],
            arms["arm_base_rawmt"],
            arms["arm_instruct_rawmt"],
        )

    frac = per_arm["arm_base_chat"]["frac_coherent"]["value"]
    if not isinstance(frac, float) or math.isnan(frac):
        raise RuntimeError("aggregate: frac_coherent(arm_base_chat) is undefined (zero kept items)")

    def _token(f: float, floor: float) -> str:
        return "coherent-as-used" if (f - floor) >= 0 else "restate-on-bare-text"

    frac_sens = per_arm["arm_base_chat"]["frac_coherent_drops_as_incoherent"]["value"]
    verdict = {
        "arm": "arm_base_chat",
        "frac_coherent": frac,
        "floor": VERDICT_FLOOR,
        "delta_frac": frac - VERDICT_FLOOR,
        "token": _token(frac, VERDICT_FLOOR),
        "sensitivity": {
            "floor_0.70": _token(frac, 0.70),
            "floor_0.90": _token(frac, 0.90),
            "drops_as_incoherent_at_0.80": {
                "frac": frac_sens,
                "token": _token(frac_sens, VERDICT_FLOOR),
            },
        },
    }
    # Strip the bulky per-item maps from the persisted per-arm block; keep them
    # in a dedicated per_item section (still needed for reproducibility +
    # the figures phase, which consumes ONLY this JSON).
    per_item = {name: stats.pop("kept_scores") for name, stats in per_arm.items()}
    per_item_d3 = {name: stats.pop("kept_d3") for name, stats in per_arm.items()}
    payload = {
        "metadata": _meta("aggregate"),
        "judge_config": {
            "n_draws": N_DRAWS_PROD,
            "max_tokens": JUDGE_MAX_TOKENS,
            "threshold_base": 0,
            "coherent_threshold": COHERENT_THRESHOLD,
            "rubric_sha256": _sha(RUBRIC),
        },
        "arms": per_arm,
        "paired_deltas": pairs,
        "verdict": verdict,
        "per_item_mean_scores": per_item,
        "per_item_distinct_3gram": per_item_d3,
    }
    _write_json(out_json, payload)
    return payload


def _decsens_verdict_token(frac: float, floor: float) -> str:
    """Plan v5 §3 lattice: render-and-sampling <=> (frac - floor) >= 0; render-driven else."""
    return "render-and-sampling" if (frac - floor) >= 0 else "render-driven"


def _parent_item_means(arm: str) -> dict[int, float]:
    """Item-mean judge scores for a PARENT temperature-1.0 arm, read READ-ONLY from the
    committed parent judge_raw file (plan v5 §3 row-coverage; filesystem-first with the
    git-blob fallback for sparse checkouts).

    Reproduces judge_result_from_save_raw's kept-draw semantics exactly: a draw is KEPT iff
    graded_judge._score_from_parsed(parsed) is not None (content drops, transport losses and
    api-refusals all yield None there); item mean = plain mean over kept draws; items with
    zero kept draws are OMITTED (they become excluded pairs downstream). Keys are re-based
    to prompt_idx ints parsed from the '{arm}--{prompt_idx}' item-id grammar.
    """
    from explore_persona_space.eval.graded_judge import _score_from_parsed

    raw = _read_committed_json(f"eval_results/issue_2477/judge/judge_raw_{arm}.json")
    all_scores = raw.get("all_scores") or {}
    if not all_scores:
        raise RuntimeError(f"parent judge_raw for {arm}: all_scores empty/missing")
    prefix = f"{arm}--"
    draws: dict[int, list[float]] = {}
    for cid, parsed in all_scores.items():
        item_id = cid.rsplit("__", 2)[0]
        if not item_id.startswith(prefix):
            raise RuntimeError(f"parent judge_raw for {arm}: unexpected item id {item_id!r}")
        s = _score_from_parsed(parsed)
        if s is not None:
            draws.setdefault(int(item_id[len(prefix) :]), []).append(s)
    if not draws:
        raise RuntimeError(f"parent judge_raw for {arm}: zero kept draws across all items")
    return {idx: sum(v) / len(v) for idx, v in draws.items()}


def _paired_delta_vs_parent(
    new_stats: dict, new_arm: ArmData, parent_means: dict[int, float]
) -> dict:
    """Per-prompt delta (fresh arm − parent temperature-1.0 arm) over prompt_idx pairs kept
    on BOTH sides; zero-kept pairs excluded and counted (plan v5 §3 registered contrasts).
    Same output shape as _paired_delta so downstream consumers read one schema."""
    ka = {new_arm.pair_key[i]: s for i, s in new_stats["kept_scores"].items()}
    shared = sorted(set(ka) & set(parent_means))
    deltas = [ka[k] - parent_means[k] for k in shared]
    return {
        "n_pairs": len(deltas),
        "n_excluded_pairs": new_stats["n_items"] - len(deltas),
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "delta_ci95": _boot_ci_mean(deltas),
        "per_pair_delta": {str(k): float(ka[k] - parent_means[k]) for k in shared},
    }


def _aggregate_decsens_core(
    arms: dict[str, ArmData], save_raw_paths: dict[str, Path], out_json: Path
) -> dict:
    """Plan v5 B6': per-arm stats for the 4 fresh temperature arms + the four registered
    cross-temperature paired deltas against the parent's committed judge_raw (read-only) +
    the §3 verdict lattice keyed on frac_coherent(arm_base_chat_t07) vs the 0.80 floor."""
    per_arm = {name: _arm_stats(arm, save_raw_paths[name]) for name, arm in arms.items()}
    missing_arms = [n for n in DECSENS_ARM_NAMES if n not in per_arm]
    if missing_arms:
        raise RuntimeError(f"aggregate: decoding-sensitivity arms missing: {missing_arms}")

    parent_means = {arm: _parent_item_means(arm) for arm in ("arm_base_chat", "arm_base_bare")}
    pairs = {
        key: _paired_delta_vs_parent(per_arm[new_arm], arms[new_arm], parent_means[parent_arm])
        for key, new_arm, parent_arm in DECSENS_PAIR_SPECS
    }

    frac = per_arm["arm_base_chat_t07"]["frac_coherent"]["value"]
    if not isinstance(frac, float) or math.isnan(frac):
        raise RuntimeError(
            "aggregate: frac_coherent(arm_base_chat_t07) is undefined (zero kept items)"
        )
    frac_sens = per_arm["arm_base_chat_t07"]["frac_coherent_drops_as_incoherent"]["value"]
    verdict = {
        "arm": "arm_base_chat_t07",
        "frac_coherent": frac,
        "floor": VERDICT_FLOOR,
        "delta_frac": frac - VERDICT_FLOOR,
        "token": _decsens_verdict_token(frac, VERDICT_FLOOR),
        "sensitivity": {
            "floor_0.70": _decsens_verdict_token(frac, 0.70),
            "floor_0.90": _decsens_verdict_token(frac, 0.90),
            "drops_as_incoherent_at_0.80": {
                "frac": frac_sens,
                "token": _decsens_verdict_token(frac_sens, VERDICT_FLOOR),
            },
        },
    }
    # Parent temperature-1.0 comparator context, recomputed from the SAME kept-draw item
    # means the paired deltas consume, so --phase figures stays a pure verdict-JSON consumer.
    parent_comparators = {}
    for arm, means in parent_means.items():
        n_kept = len(means)
        n_coh = sum(1 for v in means.values() if v >= COHERENT_THRESHOLD)
        parent_comparators[arm] = {
            "n_items_kept": n_kept,
            "frac_coherent": {
                "value": n_coh / n_kept,
                "wilson_ci95": _wilson_ci(n_coh, n_kept),
                "n_coherent": n_coh,
                "n_kept": n_kept,
            },
            "mean": float(np.mean(list(means.values()))),
            "source": (
                f"eval_results/issue_2477/judge/judge_raw_{arm}.json "
                "(parent round, temperature 1.0, read-only)"
            ),
        }
    per_item = {name: stats.pop("kept_scores") for name, stats in per_arm.items()}
    per_item_d3 = {name: stats.pop("kept_d3") for name, stats in per_arm.items()}
    payload = {
        "metadata": _meta("aggregate"),
        "condition_set": DECSENS,
        "judge_config": {
            "n_draws": N_DRAWS_PROD,
            "max_tokens": JUDGE_MAX_TOKENS,
            "threshold_base": 0,
            "coherent_threshold": COHERENT_THRESHOLD,
            "rubric_sha256": _sha(RUBRIC),
        },
        "arms": per_arm,
        "paired_deltas_vs_parent_t10": pairs,
        "parent_comparators": parent_comparators,
        "verdict": verdict,
        "per_item_mean_scores": per_item,
        "per_item_distinct_3gram": per_item_d3,
    }
    _write_json(out_json, payload)
    return payload


def _aggregate_smoke_fixture(scratch: Path) -> tuple[dict[str, ArmData], dict[str, Path]]:
    """Synthetic fixture exercising the aggregation math end-to-end (benign text only)."""
    scratch.mkdir(parents=True, exist_ok=True)

    def _mk_arm(name: str, n: int, depths: list[int] | None = None) -> ArmData:
        a = ArmData(name=name)
        for i in range(n):
            iid = f"{name}--{i}" if depths is None else f"{name}--conv{i}--d{depths[i]}"
            text = f"synthetic answer {i} about testing aggregation with several words " * (i + 1)
            a.items.append((iid, f"synthetic question {i}", text))
            a.pair_key[iid] = i if depths is None else (f"conv{i}", depths[i])
            if depths is not None:
                a.depth[iid] = depths[i]
            a.cap_hit[iid] = i == 0
        return a

    arms = {
        "arm_instruct_chat": _mk_arm("arm_instruct_chat", 6),
        "arm_base_chat": _mk_arm("arm_base_chat", 6),
        "arm_base_rawmt": _mk_arm("arm_base_rawmt", 4, depths=[2, 2, 3, 3]),
        "arm_instruct_rawmt": _mk_arm("arm_instruct_rawmt", 4, depths=[2, 2, 3, 3]),
    }
    arms["arm_instruct_chat"].cap_hit = {}
    arms["arm_instruct_chat"].cap_hit_note = "N/A — not recorded in the banked artifact"

    draws_by_arm: dict[str, dict[str, list[object]]] = {
        "arm_instruct_chat": {
            f"arm_instruct_chat--{i}": [{"score": 90 + (i % 3), "stop_reason": "end_turn"}, 88]
            for i in range(6)
        },
        "arm_base_chat": {
            "arm_base_chat--0": [{"score": 80, "stop_reason": "end_turn"}, 76],
            "arm_base_chat--1": [70, 72],
            "arm_base_chat--2": [{"score": 40, "stop_reason": "end_turn"}, 44],
            "arm_base_chat--3": [90, {"score": 92, "stop_reason": "end_turn"}],
            # one transport-lost draw + one kept draw
            "arm_base_chat--4": [{"error": True, "transport": True, "reason": "timeout"}, 65],
            # dropped item: both draws are instructed REFUSALs (content drops)
            "arm_base_chat--5": [
                {"score": "REFUSAL", "stop_reason": "end_turn"},
                {"score": "REFUSAL", "stop_reason": "end_turn"},
            ],
        },
        "arm_base_rawmt": {
            "arm_base_rawmt--conv0--d2": [60, 62],
            "arm_base_rawmt--conv1--d2": [55, 57],
            "arm_base_rawmt--conv2--d3": [45, 47],
            "arm_base_rawmt--conv3--d3": [30, 34],
        },
        "arm_instruct_rawmt": {
            "arm_instruct_rawmt--conv0--d2": [85, 87],
            "arm_instruct_rawmt--conv1--d2": [80, 82],
            "arm_instruct_rawmt--conv2--d3": [75, 77],
            "arm_instruct_rawmt--conv3--d3": [70, 72],
        },
    }
    save_raw_paths: dict[str, Path] = {}
    for name, per_item in draws_by_arm.items():
        all_scores: dict[str, object] = {}
        for j, (iid, draws) in enumerate(sorted(per_item.items())):
            for d, val in enumerate(draws):
                all_scores[f"{iid}__{j:05d}__{d:02d}"] = val
        path = scratch / f"judge_raw_{name}.json"
        _write_json(path, {"all_scores": all_scores})
        save_raw_paths[name] = path
    return arms, save_raw_paths


def phase_aggregate(args: argparse.Namespace) -> None:
    decsens = args.condition_set == DECSENS
    if args.smoke:
        if decsens:
            raise SystemExit(
                "aggregate --smoke is parent-only; the decoding-sensitivity aggregate path is "
                "unit-tested against the real committed parent judge_raw files "
                "(plan v5 §4 smoke blind-spot item iii)"
            )
        scratch = Path("/tmp/issue-2477-smoke/aggregate")
        _log(f"[phase=aggregate] SMOKE on synthetic fixture -> {scratch}")
        arms, save_raw_paths = _aggregate_smoke_fixture(scratch)
        payload = _aggregate_core(arms, save_raw_paths, scratch / "coherence_verdict_smoke.json")
        base = payload["arms"]["arm_base_chat"]
        expect = {
            "n_items_kept": 5,
            "transport": 1,
            "content": 2,
            "refusal_subset": 2,
            "n_pairs_chat": 5,
        }
        got = {
            "n_items_kept": base["n_items_kept"],
            "transport": base["drop_counts"]["transport"],
            "content": base["drop_counts"]["content"],
            "refusal_subset": base["drop_counts"]["content_refusal_subset"],
            "n_pairs_chat": payload["paired_deltas"]["base_chat_minus_instruct_chat"]["n_pairs"],
        }
        if got != expect:
            raise RuntimeError(f"aggregate smoke: fixture arithmetic mismatch: {got} != {expect}")
        _log(
            f"[phase=aggregate] SMOKE PASS: verdict={payload['verdict']['token']} "
            f"frac={payload['verdict']['frac_coherent']:.3f} checks={got}"
        )
        return

    verdict_path = (DECSENS_EVAL_DIR if decsens else EVAL_DIR) / "coherence_verdict.json"
    # Idempotency guard (cheap phase — guard rides the round-2 concern for consistency).
    if not args.force and verdict_path.exists():
        _log(f"[phase=aggregate] SKIP (idempotent): {verdict_path} exists — pass --force to re-run")
        return

    _log("[phase=aggregate] start")
    manifest = _load_manifest()
    arms = build_arms(manifest, condition_set=args.condition_set)
    judge_dir = (DECSENS_EVAL_DIR / "judge") if decsens else (EVAL_DIR / "judge")
    save_raw_paths = {}
    for name in arms:
        path = judge_dir / f"judge_raw_{name}.json"
        if not path.exists():
            raise RuntimeError(f"aggregate: {path} missing — run --phase judge-wave first")
        save_raw_paths[name] = path
    core = _aggregate_decsens_core if decsens else _aggregate_core
    payload = core(arms, save_raw_paths, verdict_path)
    v = payload["verdict"]
    _log(
        f"[phase=aggregate] done: verdict={v['token']} frac_coherent={v['frac_coherent']:.4f} "
        f"delta_frac={v['delta_frac']:+.4f}"
    )


# ---------------------------------------------------------------------------
# Phase: figures (plan B7)
# ---------------------------------------------------------------------------

# Plain-English arm labels (savefig_paper caller responsibility — no config slugs
# on any rendered surface; §5 plain-English condition names).
ARM_LABELS = {
    "arm_instruct_chat": "Instruct, chat template (banked)",
    "arm_base_chat": "Base, chat template (fresh)",
    "arm_base_bare": "Base, bare text (fresh)",
    "arm_base_rawmt": "Base, raw multi-turn (banked)",
    "arm_instruct_rawmt": "Instruct, raw multi-turn (banked)",
}

# Decoding-sensitivity figure labels (plan v5 §6: parent bars labeled as parent-round rows).
DECSENS_ARM_LABELS = {
    "arm_base_chat_t07": "Base, chat template, temperature 0.7 (fresh)",
    "arm_base_chat_t00": "Base, chat template, greedy (fresh)",
    "arm_base_bare_t07": "Base, bare text, temperature 0.7 (fresh)",
    "arm_base_bare_t00": "Base, bare text, greedy (fresh)",
    "arm_base_chat": "Base, chat template, temperature 1.0 (parent round)",
    "arm_base_bare": "Base, bare text, temperature 1.0 (parent round)",
}
# Hero x-order: render families grouped, parent comparator bar adjacent to its family.
DECSENS_FIG_ORDER = (
    "arm_base_chat_t07",
    "arm_base_chat_t00",
    "arm_base_chat",
    "arm_base_bare_t07",
    "arm_base_bare_t00",
    "arm_base_bare",
)


def _render_figures(payload: dict, out_dir: Path) -> list[Path]:
    """Render the B7 hero + exploratory dump from a coherence_verdict payload.

    Pure consumer of the aggregate phase's JSON (arms stats + per-item maps);
    returns every file written (png/pdf/meta per stem). One color = one arm
    across every figure; axes + ticks + legend + panel titles only (no canvas
    caption blocks / annotations).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    arms_stats = payload["arms"]
    per_item = payload["per_item_mean_scores"]
    per_d3 = payload["per_item_distinct_3gram"]
    present = [n for n in ARM_NAMES if n in arms_stats]
    if not present:
        raise RuntimeError("figures: no known arms in the verdict payload")
    arm_colors = dict(zip(ARM_NAMES, paper_palette(len(ARM_NAMES))))
    delta_color = paper_palette(8)[5]  # distinct 6th color: a delta is not an arm
    rng = np.random.default_rng(0)
    written: list[Path] = []

    def _save(fig, stem: str) -> None:
        paths = savefig_paper(fig, stem, dir=out_dir)
        written.extend(paths.values())
        plt.close(fig)

    # --- hero: arm means + bootstrap CIs + per-item strip -------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for x, name in enumerate(present):
        scores = list(per_item[name].values())
        if scores:
            jit = rng.uniform(-0.17, 0.17, size=len(scores))
            ax.scatter(
                np.full(len(scores), float(x)) + jit,
                scores,
                s=11,
                alpha=0.35,
                color=arm_colors[name],
                linewidths=0,
                zorder=2,
            )
        mean = arms_stats[name]["mean"]
        lo, hi = arms_stats[name]["mean_ci95"]
        if mean is not None and not (math.isnan(lo) or math.isnan(hi)):
            ax.errorbar(
                [x],
                [mean],
                yerr=[[mean - lo], [hi - mean]],
                fmt="D",
                color=arm_colors[name],
                markersize=7,
                capsize=4,
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=5,
            )
    ax.axhline(
        COHERENT_THRESHOLD,
        ls="--",
        lw=1.0,
        color="0.45",
        label=f"coherent threshold ({int(COHERENT_THRESHOLD)})",
        zorder=1,
    )
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([ARM_LABELS[n] for n in present], rotation=18, ha="right")
    ax.set_ylabel("Coherence score (judge, 0–100)")
    ax.set_ylim(-3, 103)
    ax.set_title("Coherence by arm: mean, bootstrap 95% CI, per-item scores")
    ax.legend(loc="lower left")
    _save(fig, "coherence_by_arm")

    # --- per-arm item-mean histograms ---------------------------------------
    fig, axes = plt.subplots(
        1, len(present), figsize=(2.5 * len(present), 2.7), sharey=True, sharex=True
    )
    for ax, name in zip(np.atleast_1d(axes), present):
        scores = list(per_item[name].values())
        ax.hist(scores, bins=np.linspace(0, 100, 21), color=arm_colors[name])
        ax.set_title(ARM_LABELS[name], fontsize=8)
        ax.set_xlabel("Item-mean score")
    np.atleast_1d(axes)[0].set_ylabel("Items")
    _save(fig, "item_mean_hist_by_arm")

    # --- per-depth mean lines (raw multi-turn arms) --------------------------
    depth_arms = [
        n
        for n in ("arm_base_rawmt", "arm_instruct_rawmt")
        if arms_stats.get(n, {}).get("per_depth_mean")
    ]
    if depth_arms:
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        for name in depth_arms:
            pd_mean = arms_stats[name]["per_depth_mean"]
            depths = sorted(int(d) for d in pd_mean)
            ax.plot(
                depths,
                [pd_mean[str(d)] for d in depths],
                marker="o",
                color=arm_colors[name],
                label=ARM_LABELS[name],
            )
        ax.set_xlabel("Conversation depth (turn index)")
        ax.set_ylabel("Mean coherence score")
        ax.set_title("Coherence by depth, raw multi-turn arms")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend()
        _save(fig, "per_depth_lines_rawmt")

    # --- cap-hit bar (recorded arms only; N/A arms omitted, never a zero bar) -
    cap_arms = [n for n in present if isinstance(arms_stats[n]["cap_hit_fraction"], int | float)]
    if cap_arms:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.bar(
            range(len(cap_arms)),
            [float(arms_stats[n]["cap_hit_fraction"]) for n in cap_arms],
            color=[arm_colors[n] for n in cap_arms],
        )
        ax.set_xticks(range(len(cap_arms)))
        ax.set_xticklabels([ARM_LABELS[n] for n in cap_arms], rotation=18, ha="right")
        ax.set_ylabel("Cap-hit fraction")
        ax.set_title("Fraction ending at the 1,024-token cap")
        _save(fig, "cap_hit_by_arm")

    # --- distinct-3gram vs judge score scatter -------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name in present:
        d3_map = per_d3.get(name, {})
        xs = [d3_map[i] for i in sorted(d3_map) if i in per_item[name]]
        ys = [per_item[name][i] for i in sorted(d3_map) if i in per_item[name]]
        if xs:
            ax.scatter(
                xs,
                ys,
                s=14,
                alpha=0.5,
                color=arm_colors[name],
                linewidths=0,
                label=ARM_LABELS[name],
            )
    ax.set_xlabel("Distinct 3-gram rate (per item)")
    ax.set_ylabel("Item-mean coherence score")
    ax.set_title("Repetition companion vs judge score")
    ax.legend()
    _save(fig, "distinct3gram_vs_score")

    # --- paired-delta histogram (base chat minus instruct chat) --------------
    pair = payload.get("paired_deltas", {}).get("base_chat_minus_instruct_chat")
    if pair and pair.get("per_pair_delta"):
        deltas = list(pair["per_pair_delta"].values())
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        ax.hist(deltas, bins=21, color=delta_color)
        ax.set_xlabel("Coherence delta per prompt (base chat − instruct chat)")
        ax.set_ylabel("Prompts")
        ax.set_title("Paired coherence delta, shared prompt panel")
        _save(fig, "paired_delta_hist_base_chat")

    for p in written:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"figures: written file missing/empty: {p}")
    return written


def _render_figures_decsens(payload: dict, out_dir: Path) -> list[Path]:
    """Render the decoding-sensitivity hero + exploratory dump (plan v5 §6) from the
    decoding-sensitivity coherence_verdict payload.

    Pure consumer of that JSON (fresh-arm stats + parent_comparators + registered paired
    deltas); stems carry the decoding_sensitivity_ prefix; one color = one arm across every
    figure; axes + ticks + legend + panel titles only (no canvas caption blocks).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    arms_stats = payload["arms"]
    per_item = payload["per_item_mean_scores"]
    per_d3 = payload["per_item_distinct_3gram"]
    parents = payload["parent_comparators"]
    missing = [n for n in DECSENS_ARM_NAMES if n not in arms_stats]
    if missing:
        raise RuntimeError(f"figures: decoding-sensitivity arms missing from payload: {missing}")
    palette = paper_palette(8)
    colors = dict(zip(DECSENS_FIG_ORDER, palette[:6]))
    delta_color = palette[6]  # deltas are not arms — distinct 7th color
    rng = np.random.default_rng(0)
    written: list[Path] = []

    def _save(fig, stem: str) -> None:
        paths = savefig_paper(fig, stem, dir=out_dir)
        written.extend(paths.values())
        plt.close(fig)

    def _frac_block(name: str) -> dict:
        return (arms_stats[name] if name in arms_stats else parents[name])["frac_coherent"]

    # --- hero: frac_coherent bars (Wilson CI) + 0.80 floor + fresh-arm item strip ----------
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for x, name in enumerate(DECSENS_FIG_ORDER):
        fb = _frac_block(name)
        lo, hi = fb["wilson_ci95"]
        ax.bar([x], [fb["value"]], color=colors[name], width=0.62, zorder=2)
        ax.errorbar(
            [x],
            [fb["value"]],
            yerr=[[fb["value"] - lo], [hi - fb["value"]]],
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=4,
            zorder=4,
        )
    ax.axhline(
        VERDICT_FLOOR,
        ls="--",
        lw=1.0,
        color="0.35",
        label=f"verdict floor ({VERDICT_FLOOR})",
        zorder=3,
    )
    ax2 = ax.twinx()
    for x, name in enumerate(DECSENS_FIG_ORDER):
        if name in per_item:
            scores = list(per_item[name].values())
            if scores:
                jit = rng.uniform(-0.16, 0.16, size=len(scores))
                ax2.scatter(
                    np.full(len(scores), float(x)) + jit,
                    scores,
                    s=9,
                    alpha=0.3,
                    color=colors[name],
                    linewidths=0,
                    zorder=5,
                )
    ax2.set_ylabel("Item-mean judge score (0–100), fresh arms")
    ax2.set_ylim(-3, 103)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(len(DECSENS_FIG_ORDER)))
    ax.set_xticklabels([DECSENS_ARM_LABELS[n] for n in DECSENS_FIG_ORDER], rotation=22, ha="right")
    ax.set_ylabel("Fraction coherent (item-mean ≥ 50), Wilson 95% CI")
    ax.set_title("Coherence by render × temperature")
    ax.legend(loc="upper left")
    _save(fig, "decoding_sensitivity_coherence_by_arm")

    # --- per-arm item-mean histograms (fresh arms) ---------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.7), sharey=True, sharex=True)
    for ax_i, name in zip(np.atleast_1d(axes), DECSENS_ARM_NAMES):
        ax_i.hist(list(per_item[name].values()), bins=np.linspace(0, 100, 21), color=colors[name])
        ax_i.set_title(DECSENS_ARM_LABELS[name], fontsize=7)
        ax_i.set_xlabel("Item-mean score")
    np.atleast_1d(axes)[0].set_ylabel("Items")
    _save(fig, "decoding_sensitivity_item_mean_hist_by_arm")

    # --- cap-hit bar (fresh arms; recorded-numeric only, N/A omitted never zero) ----------
    cap_arms = [
        n for n in DECSENS_ARM_NAMES if isinstance(arms_stats[n]["cap_hit_fraction"], int | float)
    ]
    if cap_arms:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.bar(
            range(len(cap_arms)),
            [float(arms_stats[n]["cap_hit_fraction"]) for n in cap_arms],
            color=[colors[n] for n in cap_arms],
        )
        ax.set_xticks(range(len(cap_arms)))
        ax.set_xticklabels([DECSENS_ARM_LABELS[n] for n in cap_arms], rotation=22, ha="right")
        ax.set_ylabel("Cap-hit fraction")
        ax.set_title("Fraction ending at the 1,024-token cap (fresh arms)")
        _save(fig, "decoding_sensitivity_cap_hit_by_arm")

    # --- distinct-3gram vs judge score scatter (fresh arms) ------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name in DECSENS_ARM_NAMES:
        d3_map = per_d3.get(name, {})
        xs = [d3_map[i] for i in sorted(d3_map) if i in per_item[name]]
        ys = [per_item[name][i] for i in sorted(d3_map) if i in per_item[name]]
        if xs:
            ax.scatter(
                xs,
                ys,
                s=14,
                alpha=0.5,
                color=colors[name],
                linewidths=0,
                label=DECSENS_ARM_LABELS[name],
            )
    ax.set_xlabel("Distinct 3-gram rate (per item)")
    ax.set_ylabel("Item-mean coherence score")
    ax.set_title("Repetition companion vs judge score")
    ax.legend()
    _save(fig, "decoding_sensitivity_distinct3gram_vs_score")

    # --- cross-temperature paired-delta histograms (4 panels, plan v5 §3 contrasts) -------
    pairs = payload["paired_deltas_vs_parent_t10"]
    panel_titles = {
        "chat_t07_minus_chat_t10": "Chat: 0.7 − 1.0",
        "chat_t00_minus_chat_t10": "Chat: greedy − 1.0",
        "bare_t07_minus_bare_t10": "Bare: 0.7 − 1.0",
        "bare_t00_minus_bare_t10": "Bare: greedy − 1.0",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0), sharex=True, sharey=True)
    for ax_i, (key, title) in zip(axes.ravel(), panel_titles.items()):
        deltas = list(pairs[key]["per_pair_delta"].values())
        if deltas:
            ax_i.hist(deltas, bins=21, color=delta_color)
        ax_i.set_title(title, fontsize=9)
    for ax_i in axes[-1]:
        ax_i.set_xlabel("Per-prompt coherence delta")
    for ax_i in axes[:, 0]:
        ax_i.set_ylabel("Prompts")
    fig.suptitle("Cross-temperature paired deltas vs parent (temperature 1.0)")
    _save(fig, "decoding_sensitivity_paired_delta_hists")

    # --- arm means: bootstrap CIs (fresh) + parent means as points ------------------------
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for x, name in enumerate(DECSENS_FIG_ORDER):
        if name in arms_stats:
            scores = list(per_item[name].values())
            if scores:
                jit = rng.uniform(-0.17, 0.17, size=len(scores))
                ax.scatter(
                    np.full(len(scores), float(x)) + jit,
                    scores,
                    s=11,
                    alpha=0.35,
                    color=colors[name],
                    linewidths=0,
                    zorder=2,
                )
            mean = arms_stats[name]["mean"]
            lo, hi = arms_stats[name]["mean_ci95"]
            if mean is not None and not (math.isnan(lo) or math.isnan(hi)):
                ax.errorbar(
                    [x],
                    [mean],
                    yerr=[[mean - lo], [hi - mean]],
                    fmt="D",
                    color=colors[name],
                    markersize=7,
                    capsize=4,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    zorder=5,
                )
        else:
            ax.scatter(
                [x],
                [parents[name]["mean"]],
                marker="o",
                s=48,
                color=colors[name],
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )
    ax.axhline(
        COHERENT_THRESHOLD,
        ls="--",
        lw=1.0,
        color="0.45",
        label=f"coherent threshold ({int(COHERENT_THRESHOLD)})",
        zorder=1,
    )
    ax.set_xticks(range(len(DECSENS_FIG_ORDER)))
    ax.set_xticklabels([DECSENS_ARM_LABELS[n] for n in DECSENS_FIG_ORDER], rotation=22, ha="right")
    ax.set_ylabel("Coherence score (judge, 0–100)")
    ax.set_ylim(-3, 103)
    ax.set_title("Arm means: bootstrap 95% CI (fresh), parent means as points")
    ax.legend(loc="lower left")
    _save(fig, "decoding_sensitivity_arm_means")

    for p in written:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"figures: written file missing/empty: {p}")
    return written


def phase_figures(args: argparse.Namespace) -> None:
    """Plan B7 / v5 §6: render figures off the aggregate outputs (verdict JSON only)."""
    decsens = args.condition_set == DECSENS
    if args.smoke:
        if decsens:
            raise SystemExit(
                "figures --smoke is parent-only; the decoding-sensitivity figures phase is a "
                "cheap VM-side pure consumer of the decoding-sensitivity verdict JSON "
                "(plan v5 §4)"
            )
        scratch = Path("/tmp/issue-2477-smoke/figures")
        _log(f"[phase=figures] SMOKE: synthetic fixture -> {scratch} (no canonical writes)")
        arms, save_raw_paths = _aggregate_smoke_fixture(scratch / "fixture")
        payload = _aggregate_core(
            arms, save_raw_paths, scratch / "fixture" / "coherence_verdict_smoke.json"
        )
        out_dir = scratch
    else:
        # Idempotency guard (cheap phase — rides the round-2 concern for consistency).
        hero_name = (
            "decoding_sensitivity_coherence_by_arm.png" if decsens else "coherence_by_arm.png"
        )
        hero = REPO_ROOT / "figures" / "issue_2477" / hero_name
        if not args.force and hero.exists():
            _log(f"[phase=figures] SKIP (idempotent): {hero} exists — pass --force to re-run")
            return
        verdict_path = (DECSENS_EVAL_DIR if decsens else EVAL_DIR) / "coherence_verdict.json"
        if not verdict_path.exists():
            raise RuntimeError(
                "figures: run --phase aggregate first (coherence_verdict.json missing)"
            )
        payload = json.loads(verdict_path.read_text(encoding="utf-8"))
        out_dir = REPO_ROOT / "figures" / "issue_2477"
    written = (
        _render_figures_decsens(payload, out_dir) if decsens else _render_figures(payload, out_dir)
    )
    stems = sorted({p.stem for p in written if p.suffix == ".png"})
    _log(f"[phase=figures] done: {len(written)} files ({len(stems)} figures) -> {out_dir}")
    _log(f"[figures] stems: {stems}")


PHASES = {
    "inventory": phase_inventory,
    "sample": phase_sample,
    "gen": phase_gen,
    "judge-pilot": phase_judge_pilot,
    "judge-wave": phase_judge_wave,
    "aggregate": phase_aggregate,
    "figures": phase_figures,
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="#2477 base-coherence driver (plan v4)")
    ap.add_argument("--phase", choices=sorted(PHASES), help="phase to run")
    ap.add_argument(
        "--condition-set",
        choices=sorted(CONDITION_SETS),
        default="parent",
        help=(
            "condition-set registry key (plan v5 C0); default parent = the parent round's "
            "behavior, unchanged"
        ),
    )
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED_DEFAULT, help="sample-phase seed")
    ap.add_argument("--smoke", action="store_true", help="tiny-slice smoke mode (n only)")
    ap.add_argument(
        "--out",
        default="/workspace/results/issue_2477",
        help="gen-phase out root (pod-side; plan phase_outputs)",
    )
    ap.add_argument("--import-check", action="store_true", help="static args/bind check, then exit")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run past a phase's idempotency guard (default off: completed phases skip loud)",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check)")
    PHASES[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    if args.phase == "gen":
        # vLLM generation driver terminal: finalization-skipping exit AFTER verified
        # uploads + sentinel write (gotchas.md: sys.exit is not a terminal for vLLM).
        os._exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
