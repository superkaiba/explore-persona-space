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
- ``figures``     UNIT-2 stub (pre-split build; lands in the follow-up unit).

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
    for path, size in files:
        name = Path(path).name
        if not _is_meta_sidecar(name) or not name.lower().endswith(".json"):
            continue
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
    """Classify one completion-bank candidate: model + render + provenance + evidence."""
    model: str | None = None
    render: str | None = None
    provenance = "on-policy generated"
    evidence: list[str] = []
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
    else:
        model, render, evidence = _determine_from_sidecars(dir_payloads)
        if "/instruct/" in path and model is None:
            model = MODEL_INSTRUCT
            evidence.append("path token: /instruct/")
        if ("/pretrained/" in path or "/base/" in path) and model is None:
            model = MODEL_BASE
            evidence.append("path token: /pretrained|base/")
    is_base = bool(
        model
        and "qwen2.5-7b" in model.lower()
        and "instruct" not in model.lower()
        and render == "chat template"
    )
    return {
        "classification": "completion-bank",
        "model": model or "undetermined",
        "render": render or "undetermined",
        "provenance": provenance,
        "evidence": evidence or ["path-only classification (no sidecar evidence)"],
        "is_base_generated_chat_template_bank": is_base,
    }


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
    # Dynamic rows: any bank under the non-825 roots (or unexpected 825 subtrees).
    static_prefixes = ("/track_s/", "/turn_dynamics/", "/onpolicy_turn_depth/")
    for row in bank_rows:
        if any(tok in row["path"] for tok in static_prefixes):
            continue
        root = next((r for r in ROOTS if row["path"].startswith(r)), row["path"].split("/")[0])
        rows.append(
            {
                "task": f"root {root}",
                "artifact": row["path"],
                "generated_by_model": row["model"],
                "render": row["render"],
                "provenance": row["provenance"],
                "consuming_fit_result": "inventory evidence only (classified from listing)",
                "evidence_quote": "; ".join(row["evidence"])[:400],
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
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    from explore_persona_space.orchestrate import hub

    _log("[phase=inventory] start")
    api = HfApi()
    all_files: list[tuple[str, int]] = []
    for root in ROOTS:
        entries = hub.retry_transient(
            lambda root=root: list(
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
        name = Path(path).name
        suffix = Path(path).suffix.lower()
        row: dict = {"path": path, "size": size}
        if suffix not in (".json", ".jsonl", ".md"):
            row.update(
                classification=f"excluded:non-completion-format ({suffix or 'no-extension'})",
                model=None,
                render=None,
            )
        elif _is_meta_sidecar(name):
            row.update(classification="meta-sidecar", model=None, render=None)
        else:
            row.update(_classify_bank(path, payloads.get(str(Path(path).parent), {})))
            bank_rows.append(row)
        manifest_rows.append(row)

    candidates = [r["path"] for r in bank_rows if r.get("is_base_generated_chat_template_bank")]
    phase_c_fires_pre_parity = len(candidates) == 0
    n_by_class: dict[str, int] = {}
    for row in manifest_rows:
        key = row["classification"].split(" ")[0].split(":")[0]
        n_by_class[key] = n_by_class.get(key, 0) + 1

    _write_json(
        EVAL_DIR / "inventory_manifest.json",
        {
            "metadata": _meta("inventory"),
            "roots": ROOTS,
            "n_files": len(manifest_rows),
            "n_by_class": n_by_class,
            "unparseable_sidecars": unparseable,
            "contingency": {
                "phase_c_fires_pre_parity": phase_c_fires_pre_parity,
                "candidate_base_chat_banks": candidates,
                "note": (
                    "Phase C fires iff no file classifies as a Qwen2.5-7B base-GENERATED "
                    "chat-template completion bank; a candidate here still must pass the "
                    "A5 parity gate at --phase sample before Phase C is skipped."
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
        for repo_path in _list_armg_step_files(api, model, shard):
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
        pre.update(_load_armg_rows(api, "pretrained", [extra]))
        ins.update(_load_armg_rows(api, "instruct", [extra]))
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


def phase_gen(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("gen is pod-side (GPU) only — refuse to run without CUDA")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from explore_persona_space.orchestrate import hub

    smoke = bool(args.smoke)
    out_root = Path(args.out)
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
    arms = [
        ("base_chat", rendered_chat, ["<|im_end|>"]),
        ("base_bare", rendered_bare, ["\nUser:", "\n\nUser:"]),
    ]
    gen_stats: dict[str, dict] = {}
    suffix = "smoke" if smoke else "seed42"
    for arm, rendered, stop in arms:
        sampling = SamplingParams(
            n=1, temperature=1.0, top_p=0.95, max_tokens=1024, seed=GEN_SEED, stop=stop
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
                        "temperature": 1.0,
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
                fr: sum(1 for r in rows if r["finish_reason"] == fr)
                for fr in sorted({r["finish_reason"] for r in rows})
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
        "stop_strings": {a: s for a, _, s in arms},
    }
    if smoke:
        _write_json(out_root / "smoke" / "gen_meta_smoke.json", gen_meta)
        _log("[phase=gen] done (smoke — no upload; eyeball smoke/*.jsonl for stop behavior)")
        return
    _write_json(out_root / "gen_meta.json", gen_meta)

    # C4: upload BEFORE anything else. Explicit per-file hub._upload calls (unrolled, not a
    # loop) because the canonical upload_raw_completions_to_data_repo helper composes dests
    # as <exp>/raw_completions/<rel> with selection requiring a local raw_completions/ dir —
    # which would double the prefix vs the plan-declared destination.
    dest_prefix = f"{EXPERIMENT}/raw_completions/generation"
    chat_path = out_root / f"base_chat_{suffix}.jsonl"
    bare_path = out_root / f"base_bare_{suffix}.jsonl"
    meta_path = out_root / "gen_meta.json"
    u1 = hub._upload(
        chat_path,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{dest_prefix}/{chat_path.name}",
        upload_as_file=True,
        raise_on_error=True,
    )
    u2 = hub._upload(
        bare_path,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{dest_prefix}/{bare_path.name}",
        upload_as_file=True,
        raise_on_error=True,
    )
    u3 = hub._upload(
        meta_path,
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{dest_prefix}/{meta_path.name}",
        upload_as_file=True,
        raise_on_error=True,
    )
    if not (u1 and u2 and u3):
        raise RuntimeError("gen: one of the three uploads returned an empty url")
    from huggingface_hub import HfApi

    expected = [f"{dest_prefix}/{p.name}" for p in (chat_path, bare_path, meta_path)]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), hub.DEFAULT_DATASET_REPO, expected, path_in_repo=dest_prefix
    )
    if missing:
        raise RuntimeError(f"gen: files missing on Hub after upload: {missing}")

    # Mirror the small text files onto the issue branch tree (plan C4).
    mirror = REPO_ROOT / "eval_results" / "issue_2477" / "fresh_completions"
    mirror.mkdir(parents=True, exist_ok=True)
    for p in (chat_path, bare_path):
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


def _fresh_rows(name: str) -> list[dict]:
    path = EVAL_DIR / "fresh_completions" / name
    if not path.exists():
        raise RuntimeError(
            f"judge: {path} missing — Phase C (gen) has not produced fresh completions yet"
        )
    return _read_jsonl(path)


def build_arms(manifest: dict) -> dict[str, ArmData]:
    """Assemble the five §5 arms; item_id grammar per plan B1 ('--' delimiter)."""
    arms: dict[str, ArmData] = {}

    chat_items = manifest["chat_items"]
    prompt_by_idx = {it["prompt_idx"]: it["prompt"] for it in chat_items}
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
        a = ArmData(name=arm_name)
        for row in _fresh_rows(fname):
            p_idx = row["prompt_idx"]
            if p_idx not in prompt_by_idx:
                raise RuntimeError(
                    f"{arm_name}: fresh row prompt_idx={p_idx} not in the sampled panel"
                )
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
        arms[arm_name] = a

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
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    _log("[phase=judge-pilot] start")
    manifest = _load_manifest()
    arms = build_arms(manifest)
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    report = judge_pilot_gate(
        {name: a.items for name, a in arms.items()},
        RUBRIC,
        max_tokens=JUDGE_MAX_TOKENS,
        cache_dir=REPO_ROOT / "data" / "issue_2477" / "judge_cache_pilot" / run_ts,
        save_raw_dir=EVAL_DIR / "judge" / "pilot_raw",
        n_draws=N_DRAWS_PILOT,
        target_total_draws=PILOT_TARGET_TOTAL_DRAWS,
        parse_fail_threshold=0.02,
        min_effective_draws_per_arm=10,
        wave_threshold_base=0,
        report_path=EVAL_DIR / "judge" / "pilot_report.json",
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


def phase_judge_wave(args: argparse.Namespace) -> None:
    from explore_persona_space.eval.graded_judge import judge_graded

    from explore_persona_space.orchestrate import hub

    manifest = _load_manifest()
    run_ts = time.strftime("%Y%m%d-%H%M%S")

    if args.smoke:
        # B3: live forced-batch smoke through the run's exact request builder.
        _log("[phase=judge-wave] B3 live forced-batch smoke (5 items x 1 draw, threshold_base=0)")
        scratch = Path("/tmp/issue-2477-smoke/judge")
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

    pilot_path = EVAL_DIR / "judge" / "pilot_report.json"
    if not pilot_path.exists():
        raise RuntimeError("judge-wave: run --phase judge-pilot first (pilot_report.json missing)")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if not pilot.get("passed"):
        raise RuntimeError("judge-wave: pilot gate did not PASS — fix + re-pilot before the wave")

    arms = build_arms(manifest)
    for name, arm in arms.items():
        save_raw = EVAL_DIR / "judge" / f"judge_raw_{name}.json"
        _log(
            f"[judge-wave] arm {name}: {len(arm.items)} items x {N_DRAWS_PROD} draws (batch-pinned)"
        )
        result = judge_graded(
            arm.items,
            RUBRIC,
            n_draws=N_DRAWS_PROD,
            cache_dir=REPO_ROOT / "data" / "issue_2477" / "judge_cache" / run_ts / name,
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
        EVAL_DIR / "judge",
        hub.DEFAULT_DATASET_REPO,
        "dataset",
        f"{EXPERIMENT}/judge_raw",
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
    # in a dedicated per_item section (still needed for reproducibility).
    per_item = {name: stats.pop("kept_scores") for name, stats in per_arm.items()}
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
    if args.smoke:
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

    _log("[phase=aggregate] start")
    manifest = _load_manifest()
    arms = build_arms(manifest)
    save_raw_paths = {}
    for name in arms:
        path = EVAL_DIR / "judge" / f"judge_raw_{name}.json"
        if not path.exists():
            raise RuntimeError(f"aggregate: {path} missing — run --phase judge-wave first")
        save_raw_paths[name] = path
    payload = _aggregate_core(arms, save_raw_paths, EVAL_DIR / "coherence_verdict.json")
    v = payload["verdict"]
    _log(
        f"[phase=aggregate] done: verdict={v['token']} frac_coherent={v['frac_coherent']:.4f} "
        f"delta_frac={v['delta_frac']:+.4f}"
    )


def phase_figures(args: argparse.Namespace) -> None:
    raise SystemExit("figures phase lands in unit 2")


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
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED_DEFAULT, help="sample-phase seed")
    ap.add_argument("--smoke", action="store_true", help="tiny-slice smoke mode (n only)")
    ap.add_argument(
        "--out",
        default="/workspace/results/issue_2477",
        help="gen-phase out root (pod-side; plan phase_outputs)",
    )
    ap.add_argument("--import-check", action="store_true", help="static args/bind check, then exit")
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
