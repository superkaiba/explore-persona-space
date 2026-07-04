# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, σ, →) in scientific docstrings.
"""Issue #922 shared constants, pinned HF loaders, and pod-side helpers.

Layer/row convention (used by every #922 artifact):

- The position STORE has ``R = 29`` rows: row 0 = the EMBEDDING stream
  (``model.model.embed_tokens`` output — the hook the #922 driver adds
  EXPLICITLY; the #779 capture driver hooks blocks only), rows ``1..28`` =
  decoder-block outputs of blocks ``0..27`` (pre-final-norm at block 27 — the
  hook-path convention of ``analysis/extraction.py``).
- All PARENT artifacts (#779 ``cx.pt`` trajectories, ``r_B``) index by BLOCK
  ``0..27``; block ``b`` ≡ store row ``b + 1``. Result JSONs key layers by the
  BLOCK convention (``"emb"``, ``"0"``..``"27"``) so numbers align with the
  #779/#841 line and the pre-registered read-out layers ℓ*.

Revision pins: every ``issue779_monitoring`` read is pinned. Files present at
the #841 pin ``037fcbb2…`` use it verbatim (consistency WARN 1). Three files
POSTDATE that revision (verified 2026-07-03 via ``get_paths_info``: the lmsys
g-rollouts + the sycophancy/hallucination extraction artifacts) and are pinned
to the data-repo main commit ``699b5a86…`` + per-file sha256 asserts (#600
pattern) — recorded as concern ``lmsys-artifacts-not-at-pinned-revision``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

if Path("/workspace").is_dir():  # pod/GCE lanes only; never redirect the VM cache
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue779_common as C779  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger("issue922_common")

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = C779.DEFAULT_MODEL  # Qwen/Qwen2.5-7B-Instruct
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
GENERATION_SUFFIX = C779.GENERATION_SUFFIX  # "<|im_start|>assistant\n"
IM_END_ID = 151645
NL_ID = 198

HF_DATA_REPO = C779.HF_DATA_REPO
HF_PREFIX = "issue779_monitoring"
HF_REVISION = "037fcbb210bc52c459959b0746cc268fe08bae96"  # the #841 pin (WARN 1)
# Data-repo main sha resolved 2026-07-03 for the three files that POSTDATE the
# #841 pin (they do not exist at HF_REVISION — verified via get_paths_info).
HF_REVISION_LATE = "699b5a86cf10d2a087dac9c1d9cf29274b122b16"
# EPM922_HF_PREFIX override: the dispatcher smoke points uploads at
# issue922_nexttoken/smoke so smoke artifacts never mix into production paths.
HF_OUT_PREFIX = os.environ.get("EPM922_HF_PREFIX", "issue922_nexttoken")
HF_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"

LMSYS_ROLLOUTS_REL = "training-source-ablation-hg/lmsys_g_labels/lmsys_g_rollouts.json"
EXPECTED_SHA256 = {
    LMSYS_ROLLOUTS_REL: "6a027db30bd9ecccd58cd107a46abb9266d09559faed41f5676e9aed268f54aa",
    "artifacts/sycophancy.json": "7d5d4e073e4dfc5339ebaec2c219ad4bbe0ad1687071f18fee1c66cc40eafb1c",
    "artifacts/hallucination.json": (
        "38ea07d8e5616882b89ad87fd2af36726c61036bed2196ff2206d94380f5a5df"
    ),
}
HF_DL_DIR = PROJECT_ROOT / "data" / "issue_922" / "hf_dl"

N_LMSYS = 5000
W_P = 8
W_A = 40
SHARD_CTX = 500
SPLIT_SEED = 42
N_FIT, N_VAL, N_TEST = 4000, 500, 500
MLP_INIT_SEED = 658
BOOTSTRAP_SEED = 0
EVAL_N_PER_CELL = 16
EVAL_SUBSET_SEED = 42

TRAITS = ("evil", "sycophancy", "hallucination")
# Pre-registered read-out layers (BLOCK indices, the #779 convention).
PRIMARY_LSTAR = {"evil": 20, "sycophancy": 26, "hallucination": 17}
COMPANION_LSTAR = {"evil": 14, "sycophancy": 19, "hallucination": 24}
READOUT_BLOCKS = sorted({*PRIMARY_LSTAR.values(), *COMPANION_LSTAR.values()})  # [14,17,19,20,24,26]

# Segment tags for transitions (source position t → t+1); plan §4.2.
SEG_PROMPT, SEG_BOUNDARY, SEG_ANSWER, SEG_TEMPLATE_END = 0, 1, 2, 3
SEG_NAMES = {0: "prompt", 1: "boundary", 2: "answer", 3: "template_end"}

ROLLOUT_K_MAX = 40  # DV2 horizon cap (bounded by W_A)
READOUT_K_MAX = 32  # DV3 horizon (headline k ≤ 32)


def block_to_row(block: int | str) -> int:
    """Store-row index for a block ('emb' → 0; block b → b+1)."""
    if block == "emb":
        return 0
    return int(block) + 1


def row_to_block_key(row: int) -> str:
    """JSON layer key for a store row (0 → 'emb'; r → str(r−1))."""
    return "emb" if row == 0 else str(row - 1)


# ── pinned HF fetch ───────────────────────────────────────────────────────────


def sha256_path(p: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(rel_path: str, *, revision: str = HF_REVISION) -> Path:
    """hf_hub_download one file under HF_PREFIX at a pinned revision.

    Fail-loud; asserts the EXPECTED_SHA256 pin when one is registered for
    ``rel_path`` (covers already-cached copies too — the trust boundary is the
    read, not the download).
    """
    from huggingface_hub import hf_hub_download

    HF_DL_DIR.mkdir(parents=True, exist_ok=True)
    p = Path(
        hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=f"{HF_PREFIX}/{rel_path}",
            repo_type="dataset",
            revision=revision,
            cache_dir=str(HF_DL_DIR),
        )
    )
    want = EXPECTED_SHA256.get(rel_path)
    if want is not None:
        got = sha256_path(p)
        assert got == want, f"sha256 mismatch for {rel_path}: {got} != pinned {want}"
    return p


# ── LMSYS items (fit corpus) ──────────────────────────────────────────────────


def load_lmsys_items(n_contexts: int | None = None) -> list[dict]:
    """LMSYS g-label rollouts → items [{ci, messages, response}] (pinned + sha).

    Mirrors ``issue779_capture_answer_summaries.build_lmsys_items`` (bare user
    prompt, no system message; exactly 1 persisted on-policy completion per
    context; ``n_contexts==5000`` asserted) with the fetch pinned to
    ``HF_REVISION_LATE`` + the sha256 assert. ``n_contexts`` truncates for the
    smoke (the first N contexts in ci order).
    """
    path = _fetch(LMSYS_ROLLOUTS_REL, revision=HF_REVISION_LATE)
    with open(path) as f:
        blob = json.load(f)
    rollouts = blob["rollouts"]
    n = blob.get("n_contexts", len(rollouts))
    if len(rollouts) != n or n != N_LMSYS:
        raise RuntimeError(f"lmsys rollouts: {len(rollouts)} rows, n_contexts={n} != {N_LMSYS}")
    upto = n if n_contexts is None else min(n, n_contexts)
    items = []
    for ci in range(upto):
        row = rollouts[str(ci)]
        comps = row["responses"]
        assert len(comps) == 1, (ci, len(comps))
        items.append(
            {
                "ci": ci,
                "messages": [{"role": "user", "content": row["prompt"]}],
                "response": comps[0],
            }
        )
    return items


def lmsys_dup_report(items: list[dict], split: dict) -> dict:
    """Exact + casefold-normalized duplicate LMSYS prompt counts across splits.

    Bounds within-family interpolation inflation on in-corpus curves (plan
    §4.4 diagnostics). Returns counts of test-context prompts whose exact /
    normalized text also appears among fit∪val prompts.
    """

    def _key(s: str) -> str:
        return s.strip()

    def _norm(s: str) -> str:
        return " ".join(s.casefold().split())

    prompts = {it["ci"]: it["messages"][0]["content"] for it in items}
    fitval = {int(ci) for ci in np.concatenate([split["fit"], split["val"]])}
    test = [int(ci) for ci in split["test"] if int(ci) in prompts]
    exact = {_key(prompts[ci]) for ci in fitval if ci in prompts}
    norm = {_norm(prompts[ci]) for ci in fitval if ci in prompts}
    n_exact = sum(1 for ci in test if _key(prompts[ci]) in exact)
    n_norm = sum(1 for ci in test if _norm(prompts[ci]) in norm)
    return {"n_test": len(test), "test_exact_dup_in_fitval": n_exact, "test_norm_dup": n_norm}


# ── eval-condition items (transfer corpus) ────────────────────────────────────


def eval_questions(trait: str) -> list[str]:
    """The 20 held-out eval questions for a trait (the pass_a question axis).

    evil = the paper-verbatim artifacts committed in ``issue779_common``;
    sycophancy/hallucination = the #779-generated artifacts, fetched from the
    HF mirror (``issue779_monitoring/artifacts/<trait>.json``, pinned +
    sha256 — they postdate the #841 revision pin).
    """
    assert trait in TRAITS, trait
    if trait == "evil":
        art = C779.EVIL_ARTIFACTS
    else:
        with open(_fetch(f"artifacts/{trait}.json", revision=HF_REVISION_LATE)) as f:
            art = json.load(f)
    qs = art["eval_questions"]
    assert len(qs) >= 20, (trait, len(qs))
    return list(qs)[:20]


def eval_questions_provenance(trait: str) -> str:
    """'original' when the trait's eval questions are the pass_a capture's own.

    evil = paper-verbatim COMMITTED constants (always ``original``; the #779
    prompt-construction code is byte-identical between the pass_a capture
    commit ``fc96549e59`` and HEAD). For sycophancy/hallucination the pinned
    HF artifact documents its OWN provenance: the #779 r5 reconstruction
    (2026-07-02, one day AFTER the pass_a capture landed) lists
    ``eval_questions`` under ``reconstruction.regenerated`` — the question
    TEXT the cached ``cx.pt`` states + rollouts were captured under was lost
    with the deleted GCE instance and re-GENERATED via the Claude prompt, so
    prompts rebuilt from the artifact CANNOT reproduce the cached capture's
    token stream (crash-fix r4 root cause, att-20260703-163130: fresh-vs-
    cached parity cos_mean 0.937 / cos_min 0.709 = question-change-magnitude
    divergence on the 18 regenerated-provenance cells).
    """
    assert trait in TRAITS, trait
    if trait == "evil":
        return "original"
    with open(_fetch(f"artifacts/{trait}.json", revision=HF_REVISION_LATE)) as f:
        art = json.load(f)
    rec = art.get("reconstruction") or {}
    return "regenerated" if "eval_questions" in rec.get("regenerated", []) else "original"


def capturable_eval_conditions(trait: str) -> list[dict]:
    """The eval-context conditions whose PROMPT is reconstructable: sys* + shot0.

    #779 never persisted the vLLM-generated many-shot exemplar text, so the
    shotK>0 contexts CANNOT be rebuilt (concern
    ``manyshot-exemplars-unreconstructable``); shot0 has an empty history and
    is exactly reconstructable, as are the 8 system-prompt conditions. Reuses
    ``issue779_collect.eval_context_conditions`` verbatim and filters.
    """
    import issue779_collect as I779C

    conds = I779C.eval_context_conditions(trait)
    keep = [c for c in conds if c["mode"] == "system" or c["n_shot"] == 0]
    assert len(keep) == 9, [c["cond_id"] for c in keep]
    return keep


def build_eval_subset_items(
    n_per_cell: int = EVAL_N_PER_CELL, seed: int = EVAL_SUBSET_SEED, smoke_cells: int | None = None
) -> list[dict]:
    """Per capturable (trait, condition): n_per_cell questions × FIRST rollout.

    Question subset = ``rng(seed).choice`` over the SORTED qi keys (identical
    subset per cell — a uniform panel). Items carry the SAME message
    CONSTRUCTION the #779 pass_a capture used
    (``issue779_collect.build_eval_prompt_messages`` verbatim; shot0 ⇒ empty
    exemplar history). Whether the QUESTION TEXT itself matches the cached
    capture is per-trait (``question_provenance`` on each item): evil is
    exact (committed constants); sycophancy/hallucination questions were
    post-hoc REGENERATED (see ``eval_questions_provenance``), so those cells'
    prompts differ textually from the cached ``cx.pt`` capture AND their
    (prompt question ↔ teacher-forced response) pairing is mismatched — the
    response answers the LOST original question of the same qi. Downstream
    reads restrict registered claims accordingly (concern
    ``eval-questions-regenerated-parity-rescope``). ``smoke_cells`` truncates
    to the first N cells (smoke: 1).
    """
    import issue779_collect as I779C

    cells: list[tuple[str, dict]] = []
    for trait in TRAITS:
        for cond in capturable_eval_conditions(trait):
            cells.append((trait, cond))
    if smoke_cells is not None:
        cells = cells[:smoke_cells]
    items: list[dict] = []
    gi = 0
    prov_by_trait = {t: eval_questions_provenance(t) for t in {t for t, _ in cells}}
    for trait, cond in cells:
        qs = eval_questions(trait)
        rel = f"raw_completions/{trait}_{cond['cond_id']}_seed42.json"
        with open(_fetch(rel)) as f:
            blob = json.load(f)
        assert blob["trait"] == trait and blob["condition"] == cond["cond_id"], rel
        by_qi: dict[int, str] = {}
        for rec in blob["rollouts"]:
            if rec["ri"] == 0:
                by_qi[int(rec["qi"])] = rec["response"]
        qis = sorted(by_qi)
        rng = np.random.default_rng(seed)
        pick = sorted(rng.choice(len(qis), size=min(n_per_cell, len(qis)), replace=False).tolist())
        for j in pick:
            qi = qis[j]
            messages = I779C.build_eval_prompt_messages(trait, cond, qs[qi], exemplars=[])
            items.append(
                {
                    "ci": gi,
                    "trait": trait,
                    "cond_id": cond["cond_id"],
                    "mode": cond["mode"],
                    "qi": qi,
                    "question_provenance": prov_by_trait[trait],
                    "messages": messages,
                    "response": by_qi[qi],
                }
            )
            gi += 1
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# PORTED from the unmerged `origin/issue-841` branch @ e2a73985417a
# (scripts/issue841_common.py), per artifact-reuse § unmerged-branch protocol.
# Adaptations (named in the #922 drift report): _fetch → the #922 pinned
# fetcher (same HF_REVISION); build_eval_traj_matrix additionally records the
# per-unit QUESTION INDEX (``qi``) — needed for the 432-item unit-panel
# restriction (consistency WARN 2); everything else verbatim.
# ═══════════════════════════════════════════════════════════════════════════════


def make_split(n: int, *, n_fit: int, n_val: int, n_test: int, seed: int = SPLIT_SEED) -> dict:
    """Deterministic 3-way split of the N contexts into fit / inner-val / test.

    The test set is NEVER used in any fit, λ-selection, or early-stopping
    decision. Returns ``{"fit": idx, "val": idx, "test": idx}`` int arrays.
    Clamps to N when n < requested total (smoke). Fails loud if N is too
    small to carve any test set. (Verbatim #841 ``make_split``.)
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    need = n_fit + n_val + n_test
    if n < need:
        frac_fit, frac_val = n_fit / need, n_val / need
        nf = max(1, round(n * frac_fit))
        nv = max(1, round(n * frac_val))
        nt = n - nf - nv
        assert nt >= 1, f"N={n} too small to carve a test set (fit={nf}, val={nv})"
        n_fit, n_val, n_test = nf, nv, nt
    return {
        "fit": perm[:n_fit],
        "val": perm[n_fit : n_fit + n_val],
        "test": perm[n_fit + n_val : n_fit + n_val + n_test],
    }


def list_pass_a_cells(trait: str) -> list[str]:
    """The pass_a cell ids for a trait, enumerated from the pinned listing."""
    from huggingface_hub import HfApi

    api = HfApi()
    prefix = f"{HF_PREFIX}/analysis_tensors/pass_a"
    entries = api.list_repo_tree(
        HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", revision=HF_REVISION
    )
    want = f"{prefix}/{trait}__"
    cells = sorted(
        e.path.split("pass_a/")[1][: -len("_cx.pt")]
        for e in entries
        if e.path.startswith(want) and e.path.endswith("_cx.pt")
    )
    assert cells, f"no pass_a cells found for trait {trait!r} at revision {HF_REVISION}"
    return cells


def load_rb(trait: str) -> np.ndarray:
    """Persona direction r_B for a trait, fp32 (28, 3584) — block-indexed."""
    assert trait in TRAITS, f"unknown trait {trait!r}"
    blob = torch.load(_fetch(f"r_b/{trait}.pt"), weights_only=False)
    r_b = blob["r_b"].to(torch.float32).numpy()
    assert r_b.shape == (EXPECTED_LAYERS, EXPECTED_HIDDEN), (trait, r_b.shape)
    return r_b


def load_step0() -> dict:
    """step0_oracle.json (per-layer PV/oracle within-condition r — reload-only)."""
    with open(_fetch("analysis_tensors/step0/step0_oracle.json")) as f:
        return json.load(f)


def load_eval_cells(trait: str) -> list[dict]:
    """Load a trait's pass_a cells (JSON scalars + c_x trajectories), fp32."""
    cells = []
    for cell_id in list_pass_a_cells(trait):
        with open(_fetch(f"analysis_tensors/pass_a/{cell_id}.json")) as f:
            cell = json.load(f)
        cx = torch.load(_fetch(f"analysis_tensors/pass_a/{cell_id}_cx.pt"), weights_only=True)
        layers = list(cx["layers"])
        assert layers == list(range(EXPECTED_LAYERS)), f"{cell_id} layers != range(28): {layers}"
        cx_last = cx["cx_last"].to(torch.float32).numpy()  # (n_q, 28, H)
        assert cx_last.shape[1:] == (EXPECTED_LAYERS, EXPECTED_HIDDEN), (cell_id, cx_last.shape)
        cell["_cx_last"] = cx_last
        cell["_layers"] = layers
        cells.append(cell)
    return cells


def _score_for(cell: dict, qi: int, ri: int) -> float | None:
    """Resolve a rollout's judge score from the cell's {custom_id: score} map.

    (Verbatim #841/#779 ``_score_for`` — the per-question aggregation below
    must match #779's exactly; the rig-validation self-check depends on it.)
    """
    for cid, s in cell["judge_scores"].items():
        parts = cid.split("__")
        if len(parts) < 3:
            continue
        try:
            idx, ci = int(parts[-2]), int(parts[-1])
        except ValueError:
            continue
        if idx == qi and ci == ri:
            return s
    return None


def build_eval_traj_matrix(cells: list[dict]) -> dict:
    """Per-(condition, question) trajectory matrix (faithful #779 aggregation).

    Verbatim #841 ``build_eval_traj_matrix`` with ONE added output field:
    ``qi`` (N_q,) int — the question index per unit, required for the
    432-item unit-panel restriction (WARN 2). The unit set, y aggregation
    (mean of valid rollout scores), prune rules, cond/mode assignment are
    UNCHANGED.
    """
    layers = cells[0]["_layers"]
    traj, y, cond, mode, qis = [], [], [], [], []
    cond_map: dict[str, int] = {}
    for cell in cells:
        cid = cell["cond_id"]
        cond_map.setdefault(cid, len(cond_map))
        by_q: dict[int, list[dict]] = {}
        for rec in cell["rollouts"]:
            if rec.get("empty"):
                continue
            by_q.setdefault(rec["qi"], []).append(rec)
        for qi, recs in by_q.items():
            q_scores = [s for r in recs if (s := _score_for(cell, qi, r["ri"])) is not None]
            if not q_scores:
                continue
            traj.append(cell["_cx_last"][qi, :, :])  # (28, H)
            y.append(float(np.mean(q_scores)))
            cond.append(cond_map[cid])
            mode.append(cell["mode"])
            qis.append(int(qi))
    return {
        "traj": np.array(traj, dtype=np.float32),  # (N_q, 28, H)
        "y": np.array(y, dtype=np.float64),
        "cond": np.array(cond, dtype=int),
        "mode": np.array(mode, dtype=object),
        "qi": np.array(qis, dtype=int),
        "cond_ids": list(cond_map.keys()),
        "layers": layers,
    }


def group_by_condition(
    x: np.ndarray, y: np.ndarray, cond: np.ndarray, mode: np.ndarray, which_mode: str
) -> tuple[list, list]:
    """Split (x, y) into per-condition arrays for one elicitation mode.

    (Verbatim #841/#779 ``_group_by_condition``.)
    """
    cx, cy = [], []
    sel = np.array([m == which_mode for m in mode])
    if not sel.any():
        return cx, cy
    for c in np.unique(cond[sel]):
        m = sel & (cond == c)
        cx.append(x[m])
        cy.append(y[m])
    return cx, cy


# ── reproducibility + atomic JSON (reused from issue779_common) ───────────────

reproducibility_metadata = C779.reproducibility_metadata
write_json_atomic = C779.write_json_atomic


# ── store IO ──────────────────────────────────────────────────────────────────


def shard_path(out_dir: Path, corpus: str, k: int) -> Path:
    """Canonical shard path: <out>/<corpus>/shard_<k:03d>.pt."""
    return out_dir / corpus / f"shard_{k:03d}.pt"


def load_store(store_dir: Path, corpus: str) -> dict:
    """Load all shards of one corpus into contiguous arrays — TWO-PASS.

    Pass A reads each shard, keeps only the small per-context metadata (shapes,
    token ids, segments), and DROPS the h tensors; pass B re-reads each shard
    and copies h into the preallocated contiguous ``(R, P, H)`` fp16 array.
    Peak host RAM ≈ store + ONE shard (~5 GB), not 2× store — the r1
    code-review's ``a100-40-rung-memory-unsupported`` fix (the old single-pass
    load held every shard's tensors AND the contiguous copy simultaneously,
    ~2× store ≈ 100 GB at the §9 bound). Cost: one extra disk read of the
    shard set (~minutes).

    Returns ``{"h": (R, P, H) fp16 torch (cpu), "blocks": [...], "ctx_ids":
    [...], "pos_lo": (n_ctx,), "n_pos": (n_ctx,), "prompt_len": (n_ctx,),
    "ans_len": (n_ctx,), "window_start": (n_ctx,), "segments": (P,) int16,
    "token_ids": (P,) int32, "meta": {ci: item-meta}}`` where positions of
    context i occupy ``pos_lo[i] : pos_lo[i]+n_pos[i]``.
    """
    d = store_dir / corpus
    shards = sorted(d.glob("shard_*.pt"))
    assert shards, f"no shards under {d}"
    blocks = None
    window = None
    metas: dict[int, dict] = {}
    npos_by_ci: dict[int, int] = {}
    for sp in shards:  # pass A: metadata only, h dropped shard-by-shard
        blob = torch.load(sp, weights_only=False)
        if blocks is None:
            blocks = blob["blocks"]
            window = blob.get("window")  # {"wp", "wa"} — part of the fit regime
        else:
            assert blocks == blob["blocks"], (sp, blocks, blob["blocks"])
            assert window == blob.get("window"), (sp, window, blob.get("window"))
        for ci, rec in blob["contexts"].items():
            npos_by_ci[int(ci)] = int(rec["h"].shape[0])
            metas[int(ci)] = {k: v for k, v in rec.items() if k != "h"}
        H = blob["contexts"][next(iter(blob["contexts"]))]["h"].shape[-1]
        del blob
    ctx_ids = sorted(metas)
    R = len(blocks)
    n_pos = np.array([npos_by_ci[ci] for ci in ctx_ids], dtype=np.int64)
    pos_lo = np.concatenate([[0], np.cumsum(n_pos)[:-1]])
    P = int(n_pos.sum())
    slot = {ci: i for i, ci in enumerate(ctx_ids)}
    h = torch.empty((R, P, H), dtype=torch.float16)
    token_ids = torch.empty(P, dtype=torch.int32)
    seg_all = np.full(P, -1, dtype=np.int16)  # per SOURCE position; last pos of each ctx = -1
    for sp in shards:  # pass B: copy h in, one shard resident at a time
        blob = torch.load(sp, weights_only=False)
        for ci_s, rec in blob["contexts"].items():
            i = slot[int(ci_s)]
            lo, npos = int(pos_lo[i]), int(n_pos[i])
            h[:, lo : lo + npos, :] = rec["h"].permute(1, 0, 2)  # (n_pos,R,H) → (R,n_pos,H)
            token_ids[lo : lo + npos] = rec["token_ids"]
            seg_all[lo : lo + npos - 1] = rec["segments"].astype(np.int16)
        del blob
    return {
        "h": h,
        "blocks": blocks,
        "window": window,
        "ctx_ids": ctx_ids,
        "pos_lo": pos_lo,
        "n_pos": n_pos,
        "prompt_len": np.array([metas[ci]["prompt_len"] for ci in ctx_ids]),
        "ans_len": np.array([metas[ci]["ans_len"] for ci in ctx_ids]),
        "window_start": np.array([metas[ci]["window_start"] for ci in ctx_ids]),
        "segments": seg_all,
        "token_ids": token_ids,
        "meta": metas,
    }


# ── pod-side results sentinel (poll_pipeline contract) ────────────────────────


def write_results_sentinel(note: dict, *, kind: str = "epm:results", version: int = 1) -> Path:
    """Write the end-of-run sentinel with poll_pipeline's required keys.

    ``sentinel_schema_version`` / ``kind`` / ``version`` are the
    ``_SENTINEL_REQUIRED_KEYS``; the marker body rides under ``note``. Dir
    override via ``EPM922_SENTINEL_DIR`` (VM smokes); default /workspace/logs.
    """
    d = Path(os.environ.get("EPM922_SENTINEL_DIR", "/workspace/logs"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"issue-922-{kind.replace(':', '_')}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": kind,
        "version": version,
        "task_id": 922,
        "by": "issue922_dispatch",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": json.dumps(note, default=str),
    }
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(p)
    logger.info("[sentinel] wrote %s", p)
    return p


# ── HF uploads (bulk folder commits; quota-403 overflow fallback) ─────────────


def upload_dir_bulk(
    local_dir: Path,
    path_in_repo: str,
    *,
    allow_patterns: list[str] | None = None,
    commit_message: str,
    allow_overflow: bool = True,
) -> dict:
    """ONE ``upload_folder`` commit + scoped exact-listing verify.

    Canonical data repo first; on an LFS storage-quota 403 the upload reroutes
    to the private overflow repo under the SAME path (returned event records
    the deviation for the sentinel note — the #541/#552 recovery ordering).
    Any other failure raises. Verification uses a SCOPED ``list_repo_tree``
    (the full-repo listing times out on the ~1M-file data repo).
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    local = [
        str(p.relative_to(local_dir))
        for p in sorted(local_dir.rglob("*"))
        if p.is_file() and (allow_patterns is None or any(p.match(pat) for pat in allow_patterns))
    ]
    assert local, f"nothing to upload under {local_dir} (patterns={allow_patterns})"

    def _try(repo_id: str) -> dict:
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            commit_message=commit_message,
        )
        entries = list(
            api.list_repo_tree(
                repo_id, path_in_repo=path_in_repo, repo_type="dataset", recursive=True
            )
        )
        have = {e.path[len(path_in_repo) + 1 :] for e in entries if not e.path.endswith("/")}
        missing = [f for f in local if f not in have]
        if missing:
            raise RuntimeError(f"upload verify FAILED on {repo_id}: missing {missing[:5]}...")
        return {"repo": repo_id, "n_files": len(local), "path_in_repo": path_in_repo}

    try:
        return _try(HF_DATA_REPO)
    except HfHubHTTPError as e:
        quota = e.response is not None and e.response.status_code == 403
        if not (quota and allow_overflow):
            raise
        logger.warning("[upload] canonical 403 (storage quota) — rerouting to overflow repo")
        api.create_repo(HF_OVERFLOW_REPO, repo_type="dataset", private=True, exist_ok=True)
        ev = _try(HF_OVERFLOW_REPO)
        ev["overflow"] = True
        return ev
