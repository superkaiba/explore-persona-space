"""Issue #1332 shared helpers — family registry, paths, folds, reused-artifact loaders.

Function-space similarity between per-context-family fitted context->answer maps
as a leakage predictor (plan v3). This module is imported by every #1332 phase
script (bank build P0, GPU phase P1, fits P2, analysis P2+/P3) so the family
registry, fold partition, and reused-artifact pins live in exactly one place.

Families (26): the 16 #406 conditions (``i406_conditions.CONDITIONS``) + the 10
#532 instructed bystanders (``issue532_predictor_stress._instructed_bystander_panel``,
string-identity asserted against the committed ``predictors.json`` labels).

Reused leakage DV (#532 corrected-slot follow-up, git-tree in-repo):
``eval_results/issue_532/logp_slot_followup/per_cell_{trained,base}/{s}__{t}.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

logger = logging.getLogger("issue1332.common")

TASK_ID = 1332
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLUG = "map_similarity"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = f"issue{TASK_ID}_{SLUG}"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_LAYERS = 28
HIDDEN_DIM = 3584
SENTINEL_SCHEMA_VERSION = 1

# Bank + folds (plan §4.2)
BANK_SIZE = 400
N_FOLDS = 5
FOLD_SEED = 0  # KFold(5, shuffle, rs=0) — #823 convention
SPLIT_HALF_SEED = 0
BANK_FILE = "query_bank_v1.json"
REWRITES_FILE = "class_d_rewrites_v2.json"

# Generation (plan §4.3)
MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN = 4096  # prompt budget = MAX_MODEL_LEN - MAX_NEW_TOKENS (#952 load-time rule)
VALID_MIN_RESPONSE_TOKENS = 3
FAMILY_VALID_FLOOR = 0.80

# Ridge / similarity (plan §4.4-4.5)
LAYER_GRID = (5, 9, 14, 17, 20, 23, 26)  # diagnostic grid; L* frozen by layer_freeze.json
WHITENED_GATE_LAYER = 14  # #667 recipe
COSINE_532_LAYER = 21  # committed #532 covariate layer

# Reused artifacts (plan §10 pins). Content identity is asserted via sha256 of
# the in-tree bytes recorded at implementation time (check (f) — the git SHA
# pins in the plan refer to the commits that introduced these bytes).
PER_CELL_DIR = PROJECT_ROOT / "eval_results" / "issue_532" / "logp_slot_followup"
PREDICTORS_532 = PROJECT_ROOT / "eval_results" / "issue_532" / "predictors.json"
PREDICTORS_540 = PROJECT_ROOT / "eval_results" / "issue_540" / "predictors_jsrb.json"
G_474_LOC_EP1 = (
    PROJECT_ROOT / "eval_results" / "issue_474" / "cross_eval" / "loc_ep1" / "G_logprob_matrix.json"
)

STYLIZED_CIDS = ("A3", "A4", "A5")  # pirate / comedian / villain (#474 exclusion panels)

# data/issue_406 inputs consumed at P0 (VM-local; mirrored to HF at P0 — plan §4.2)
I406_DATA_DIR = PROJECT_ROOT / "data" / "issue_406"
I406_INPUT_RELPATHS = (
    "q_train_answers.json",
    "q_test_extended_50.json",
    "class_d/rewrites_v1.json",
)

# #545 OOD arm (plan §4.3 step 4)
I545_HF_PREFIX = "issue545_behavior_testbed"


# ── generic io helpers ────────────────────────────────────────────────────────


def write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write (tmp + os.replace) — checkpoint-per-phase safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reproducibility_metadata(extra: dict | None = None) -> dict:
    """Git commit + env versions + timestamp (CLAUDE.md reproducibility rule)."""
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ},
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    meta: dict = {
        "issue": TASK_ID,
        "git_commit": commit,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import torch
        import transformers

        meta["torch_version"] = str(torch.__version__)
        meta["transformers_version"] = transformers.__version__
    except Exception:
        pass
    if extra:
        meta.update(extra)
    return meta


def phase(name: str) -> None:
    """Emit a poll_pipeline-parseable ``[phase=...]`` breadcrumb."""
    print(f"[phase={name}]", flush=True)


def write_sentinel(kind: str, note: str, extra: dict | None = None) -> Path:
    """poll_pipeline-conformant sentinel for THIS task (_SENTINEL_REQUIRED_KEYS)."""
    logs_dir = Path("/workspace/logs")
    if not logs_dir.is_dir():
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
    kind_slug = kind.replace(":", "_")
    path = logs_dir / f"issue-{TASK_ID}-{kind_slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
        "kind": kind,
        "version": 1,
        "note": note,
        "task_id": TASK_ID,
        "by": "issue1332",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote sentinel %s", path)
    return path


# ── smoke-isolated output roots (smoke NEVER writes canonical paths) ──────────


def data_root(smoke: bool, override: str | None = None) -> Path:
    """Root for generated data artifacts (bank, rollouts, capture store)."""
    if override:
        return Path(override)
    if smoke:
        return Path(os.environ.get("EPM_I1332_SMOKE_ROOT", "/tmp/issue-1332-smoke"))
    return PROJECT_ROOT / "data" / f"issue_{TASK_ID}"


def results_dir(smoke: bool, override: str | None = None) -> Path:
    """eval_results dir: canonical git path in production, scratch under smoke."""
    if override:
        return Path(override)
    if smoke:
        return data_root(True) / "eval_results"
    return PROJECT_ROOT / "eval_results" / f"issue_{TASK_ID}"


def figures_dir(smoke: bool) -> Path:
    """Figure dir: canonical git path in production, scratch under smoke."""
    if smoke:
        return data_root(True) / "figures"
    return PROJECT_ROOT / "figures" / f"issue_{TASK_ID}"


# ── family registry ───────────────────────────────────────────────────────────


def instructed_panel() -> dict[str, str]:
    """The 10 #532 instructed-bystander system prompts, imported verbatim.

    Imports the private ``_instructed_bystander_panel`` from the frozen #532
    script so the strings can never drift from the committed source (plan §4.1
    string-identity requirement).
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from issue532_predictor_stress import _instructed_bystander_panel

    return _instructed_bystander_panel()


def family_labels() -> tuple[list[str], list[str]]:
    """(sources_16, targets_26) label lists in the committed #532 panel order.

    Reads the committed ``predictors.json`` label lists (the canonical axis
    order every reused matrix uses) and asserts they match the code-side
    registries: sources == the 16 active #406 condition cids, targets ==
    sources + the 10 instructed panel labels.
    """
    from explore_persona_space.experiments.i406_conditions import CONDITIONS

    payload = json.loads(PREDICTORS_532.read_text())
    sources: list[str] = list(payload["sources"])
    targets: list[str] = list(payload["bystanders"])
    cids = [c.cid for c in CONDITIONS]
    assert set(sources) == set(cids), (sorted(sources), sorted(cids))
    assert len(sources) == 16 and len(targets) == 26, (len(sources), len(targets))
    panel = instructed_panel()
    assert set(targets) - set(sources) == set(panel.keys()), (
        "instructed panel labels drifted from predictors.json bystanders",
        sorted(set(targets) - set(sources)),
        sorted(panel.keys()),
    )
    return sources, targets


USER_TURN_HEADER = "<|im_start|>user\n"
TEMPLATE_END_TEXT = "<|im_end|>\n"  # the two template-end tokens (#779 v_mean convention)
GENERATION_SUFFIX = "<|im_start|>assistant\n"  # cx_last position assert (#594 control)


def render_family_prompt(
    family: str,
    question: str,
    tokenizer,
    class_d_rewrites: dict[str, dict[str, str]] | None,
    panel: dict[str, str],
) -> tuple[str, int]:
    """Render T_c(q) for one of the 26 families; return (prompt_text, prefix_char_end).

    Ordinary #406 cids delegate to ``build_prompt_for_condition`` (byte-exact
    train<->eval parity with #474/#532); instructed labels use the #532
    ``[system, user]`` chat-template shape. ``prefix_char_end`` is the char
    index where the prefix segment ends: everything before the user query
    content (system block + the ``<|im_start|>user\\n`` header) — the
    chat-template preamble position for B/C1/D where the prefix carries no
    family content (plan §4.3, degeneracy stated in §4.4).
    """
    from explore_persona_space.experiments.i406_conditions import (
        CONDITIONS_BY_ID,
        build_prompt_for_condition,
    )

    if family in panel:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": panel[family]},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    elif family in CONDITIONS_BY_ID:
        prompt = build_prompt_for_condition(
            CONDITIONS_BY_ID[family], question, tokenizer, class_d_rewrites=class_d_rewrites
        )
    else:
        raise KeyError(f"unknown family {family!r}")
    idx = prompt.rindex(USER_TURN_HEADER)
    prefix_char_end = idx + len(USER_TURN_HEADER)
    assert prompt.endswith(GENERATION_SUFFIX), prompt[-80:]
    return prompt, prefix_char_end


# ── bank + folds ──────────────────────────────────────────────────────────────


def load_bank(bank_path: Path) -> list[str]:
    """Load the shared query bank; assert count + non-empty strings."""
    payload = json.loads(bank_path.read_text())
    qs = payload["questions"]
    assert all(isinstance(q, str) and q.strip() for q in qs), "empty question in bank"
    assert len(qs) == len(set(qs)), "duplicate questions in bank"
    return list(qs)


def load_rewrites(path: Path) -> dict[str, dict[str, str]]:
    """Load the {question: {register: rewrite}} Class-D map (v2 bank rewrites)."""
    return json.loads(path.read_text())


def query_folds(n_queries: int) -> list[list[int]]:
    """ONE query-indexed 5-fold partition shared across ALL families (plan §4.2).

    KFold(5, shuffle, random_state=0) over range(n_queries). Returns the list
    of val-index lists, fold order fixed.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    return [sorted(val.tolist()) for _, val in kf.split(range(n_queries))]


def split_half(n_queries: int) -> tuple[list[int], list[int]]:
    """Query-indexed A/B half split (rs=0), shared across families (plan §4.2)."""
    import numpy as np

    rng = np.random.default_rng(SPLIT_HALF_SEED)
    perm = rng.permutation(n_queries)
    half = n_queries // 2
    return sorted(perm[:half].tolist()), sorted(perm[half:].tolist())


# ── reused-artifact loaders (analysis side) ───────────────────────────────────


def load_leakage_matrices() -> dict:
    """Load the reused #532 corrected-slot per-cell reads into dense matrices.

    Returns dict with:
      - ``sources`` (16), ``targets`` (26) label lists
      - ``L`` (16, 26): trained - base ``mean_logp_marker`` (PRIMARY DV)
      - ``L_margin`` (16, 26): trained - base ``mean_marker_eos_margin`` (sensitivity DV)
      - ``base_prior`` (16, 26): per-cell base ``mean_logp_marker`` (baseline)
      - ``per_q_trained`` / ``per_q_base``: {(s, t): list[float]} 50-probe
        per-q ``logp_marker`` values for the r_LL split-half read
    """
    import numpy as np

    sources, targets = family_labels()
    n_s, n_t = len(sources), len(targets)
    L = np.full((n_s, n_t), np.nan)
    L_margin = np.full((n_s, n_t), np.nan)
    base_prior = np.full((n_s, n_t), np.nan)
    per_q_trained: dict[tuple[str, str], list[float]] = {}
    per_q_base: dict[tuple[str, str], list[float]] = {}
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            tr = json.loads((PER_CELL_DIR / "per_cell_trained" / f"{s}__{t}.json").read_text())
            ba = json.loads((PER_CELL_DIR / "per_cell_base" / f"{s}__{t}.json").read_text())
            L[i, j] = tr["summary"]["mean_logp_marker"] - ba["summary"]["mean_logp_marker"]
            L_margin[i, j] = (
                tr["summary"]["mean_marker_eos_margin"] - ba["summary"]["mean_marker_eos_margin"]
            )
            base_prior[i, j] = ba["summary"]["mean_logp_marker"]
            per_q_trained[(s, t)] = [row["logp_marker"] for row in tr["per_q"]]
            per_q_base[(s, t)] = [row["logp_marker"] for row in ba["per_q"]]
    assert not np.isnan(L).any(), "missing per-cell files"
    return {
        "sources": sources,
        "targets": targets,
        "L": L,
        "L_margin": L_margin,
        "base_prior": base_prior,
        "per_q_trained": per_q_trained,
        "per_q_base": per_q_base,
    }


def load_baseline_matrices() -> dict:
    """Committed #532 cosine (L21) + #540 RB-JS matrices on the identical panel."""
    import numpy as np

    p532 = json.loads(PREDICTORS_532.read_text())
    p540 = json.loads(PREDICTORS_540.read_text())
    sources, targets = family_labels()
    assert list(p540["sources"]) == sources and list(p540["bystanders"]) == targets, (
        "predictors_jsrb.json panel labels differ from predictors.json"
    )
    cos = np.asarray(p532["cosine_matrix"], dtype=float)
    js = np.asarray(p540["js_rb_matrix"], dtype=float)
    assert cos.shape == (16, 26) and js.shape == (16, 26), (cos.shape, js.shape)
    return {"cosine_532": cos, "js_rb_540": js, "sources": sources, "targets": targets}


def offdiag_mask(sources: list[str], targets: list[str]):
    """Boolean (16, 26) mask of the 400 off-diagonal analysis cells (s != t)."""
    import numpy as np

    mask = np.ones((len(sources), len(targets)), dtype=bool)
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            if s == t:
                mask[i, j] = False
    assert int(mask.sum()) == len(sources) * len(targets) - len(sources), int(mask.sum())
    return mask


# ── HF staging (per-file, scoped — never snapshot_download on the data repo) ──


def hf_fetch(rel_path: str, dest: Path) -> Path:
    """Fetch ``HF_PREFIX/<rel_path>`` from the data repo to ``dest`` (retried)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    dest.parent.mkdir(parents=True, exist_ok=True)
    got = retry_transient(
        lambda: hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PREFIX}/{rel_path}",
            revision="main",
        ),
        what=f"hf_fetch {rel_path}",
    )
    import shutil

    shutil.copyfile(got, dest)
    return dest


def ensure_input(local: Path, rel_path: str) -> Path:
    """Local-first -> HF-fetch -> fail-loud resolution for a pipeline input."""
    if local.exists() and local.stat().st_size > 0:
        return local
    logger.info("input %s missing locally; fetching %s/%s", local, HF_PREFIX, rel_path)
    hf_fetch(rel_path, local)
    if not local.exists() or local.stat().st_size == 0:
        raise FileNotFoundError(f"input {local} unavailable locally and on HF ({rel_path})")
    return local
