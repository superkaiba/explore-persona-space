"""#1768 shared registry: arms, adapter resolution, corpus sample, constants.

Task #1768 (capture + fit only, NO training): characterise what fine-tuning a
behavior into a context does to the context->answer map, on the ModelOrganism
fleet (plan v4). This module is the LIGHT-import shared surface consumed by
`issue1768_capture.py` (p0-p7), `issue1768_fit.py` (p8) and
`issue1768_directions.py` (p9): arm enumeration from #1481's committed verdict
manifest, adapter-subfolder resolution (plan §4.1 order), the #779 n1M corpus
sample (plan §4.2), and every cross-phase constant. Heavy imports (torch,
sibling issue registries) stay lazy inside functions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:  # script-mode sibling imports (issue1586_cells etc.)
    sys.path.insert(0, str(SCRIPTS_DIR))

ISSUE = 1768
HF_PREFIX = "issue1768_mapshift"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Plan §4.3 layers: 14 = fleet content primary (#653/#1112/#1315); 19 = #779
# scaling-curve anchor; 25 = fleet marker primary + late layer for W_U (Q4).
LAYERS: tuple[int, ...] = (14, 19, 25)
HIDDEN = 3584
N_LAYERS_FULL = 28

# Plan §4.2 corpus sample (Source: #779 scaling curve + pinned splits).
N_TRAIN = 15_000
N_VAL = 400
N_TEST = 1_000
SAMPLE_SEED = 42
FLOOR_SEED = 1768
PROMPT_TOKEN_CAP = 1024  # formatted-prompt token cap, recorded with skip counts
MAX_MODEL_LEN = 4096  # covers 1024-tok prompt + template + 2048 marker decode
MAX_NEW_CONTENT = 1024
MAX_NEW_MARKER = 2048
TF_BATCH_SIZE = 8  # issue1586_cells.G.TF_BATCH_SIZE (plan §4.4)
GEN_GPU_MEM_UTIL = 0.6  # issue1586_dispatch.CAPTURE_GPU_MEM_UTIL (HF+vLLM coexistence)

MANIFEST_HF_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m"
N1M_FITS_JSON = (
    REPO_ROOT / "eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_multilayer_fits.json"
)  # noqa: E501
VERDICT_MANIFEST = REPO_ROOT / "eval_results/issue_1481/analysis/verdict_manifest.json"

MARKER_TOKEN_ID = 83399  # " ※" leading-space form (CLAUDE.md marker rule)

BEH_KEYS = ("cas", "imp", "syc")
CTX_KEYS = ("pers", "bare", "conv", "icl")
SEEDS = (42, 137)
REGIMES = ("con", "po")
LR_BY_TAG = {"lr5e6": 5e-6, "lr1e5": 1e-5, "lr3e5": 3e-5, "lr1e4": 1e-4}

BASE_UNITS = ("base_content", "base_mk")
PILOT_ARM = "imp-pers-con-lr3e5-s42"  # plan gate 1 (the #1586-reused imp pers cell)

# ── full-FT arms (plan §4.1 USER AMENDMENT, marker 2026-07-28T20:36:21Z) ─────
# The 16 #1586 matched-install full-fine-tune cells join the fleet as
# CAPTURE-ONLY arms (method axis: LoRA vs full-FT at corpus scale). Identity =
# the #1586 `issue1586_methodgen/selection/` Hub records; checkpoints live on
# the PRIVATE overflow repo (full models — no adapter, no merge).
FT_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
FT_SELECTION_HF_PREFIX = "issue1586_methodgen/selection"
FT_CKPT_HUB_PREFIX = "issue1586"  # overflow-repo prefix (#1586 p4_persist)
FT_REUSED_CELL = "syc-pers-ft-con-s42"  # the #1112-reused cell (issue1586_cells)
FT_REUSED_SUBFOLDER = "issue1112/s3_fullft_neg/checkpoint-8"
FT_BEH_KEYS = ("cas", "imp", "mk", "syc")
FT_LR = 5e-6  # #1586 executed-grid full-FT lr (issue_1112 FT_LR/MARKER_FT_LR)

# Code-pinned selected (step, metric, in_band) per ft cell — pinned VERBATIM
# from the Hub selection records (`selection.json` step/metric/in_band fields,
# fetched + verified 2026-07-28) and RE-verified against those records at p0
# (`_probe_ft_checkpoints` fails loud on drift). The 4 mk cells are
# closest_approach fallbacks BELOW the ΔG window (in_band=False) — a #1586
# selection finding carried here as an analysis caveat, never a drop. The
# reused syc-con-s42 record carries no metric field (nan; its committed
# selection is the #1112 s3_fullft_neg record).
FT_SELECTED: dict[str, tuple[int, float, bool]] = {
    "cas-pers-ft-con-s42": (10, 0.67, True),
    "cas-pers-ft-con-s137": (12, 0.79, True),
    "cas-pers-ft-po-s42": (10, 0.72, True),
    "cas-pers-ft-po-s137": (10, 0.68, True),
    "imp-pers-ft-con-s42": (14, 0.64, True),
    "imp-pers-ft-con-s137": (12, 0.6326530612244898, True),
    "imp-pers-ft-po-s42": (12, 0.67, True),
    "imp-pers-ft-po-s137": (12, 0.6262626262626263, True),
    "mk-pers-ft-con-s42": (6, 2.1477094650268556, False),
    "mk-pers-ft-con-s137": (6, 2.4166744232177733, False),
    "mk-pers-ft-po-s42": (6, 3.148651885986328, False),
    "mk-pers-ft-po-s137": (6, 2.923549461364746, False),
    "syc-pers-ft-con-s42": (8, float("nan"), True),
    "syc-pers-ft-con-s137": (24, 0.61, True),
    "syc-pers-ft-po-s42": (6, 0.8, True),
    "syc-pers-ft-po-s137": (6, 0.76, True),
}


@dataclasses.dataclass(frozen=True)
class Arm:
    """One trained checkpoint of the 72-arm fleet (40 content + 16 marker LoRA
    + 16 full-FT; plan §4.1 amendment)."""

    arm_id: str
    kind: str  # "content" | "marker"
    beh_key: str  # cas | imp | syc | mk
    ctx_key: str  # pers | bare | conv | icl
    regime: str  # con | po
    seed: int
    lr: float
    step: int
    selection_read: float  # content: judged rate; marker: delta_logp_mean
    method: str = "lora"  # "lora" | "ft" (the amendment's method axis)


def _load_manifest() -> dict:
    return json.loads(VERDICT_MANIFEST.read_text())


def content_arms(manifest: dict | None = None) -> list[Arm]:
    """The in-band content verdict arms (plan §4.1: realized count 40)."""
    man = manifest or _load_manifest()
    out: list[Arm] = []
    for beh in BEH_KEYS:
        for ctx in CTX_KEYS:
            seeds = man["content"][beh][ctx]["seeds"]
            for seed in SEEDS:
                for regime in REGIMES:
                    entry = seeds.get(str(seed), {}).get(regime)
                    if not entry:
                        continue
                    sel = entry.get("selection") or {}
                    if not sel.get("in_band"):
                        continue
                    out.append(
                        Arm(
                            arm_id=entry["arm_id"],
                            kind="content",
                            beh_key=beh,
                            ctx_key=ctx,
                            regime=regime,
                            seed=seed,
                            lr=float(entry["lr"]),
                            step=int(sel["step"]),
                            selection_read=float(sel["rate"]),
                        )
                    )
    assert len(out) == 40, f"expected 40 in-band content arms, got {len(out)}"
    return out


def marker_arms(manifest: dict | None = None) -> list[Arm]:
    """Lowest-LR in-window marker rung per (context, regime, seed) — 16 arms."""
    man = manifest or _load_manifest()
    groups: dict[tuple[str, str, int], list[tuple[float, str, dict]]] = {}
    for arm_id, entry in man["marker"]["arms"].items():
        sel = entry.get("selection") or {}
        if not sel.get("in_window"):
            continue
        lr = LR_BY_TAG[entry["lr_key"]]
        key = (entry["ctx_key"], entry["regime"], int(entry["seed"]))
        groups.setdefault(key, []).append((lr, arm_id, entry))
    out: list[Arm] = []
    for (ctx, regime, seed), cands in sorted(groups.items()):
        lr, arm_id, entry = min(cands, key=lambda t: t[0])
        sel = entry["selection"]
        out.append(
            Arm(
                arm_id=arm_id,
                kind="marker",
                beh_key="mk",
                ctx_key=ctx,
                regime=regime,
                seed=seed,
                lr=lr,
                step=int(sel["step"]),
                selection_read=float(sel["delta_logp_mean"]),
            )
        )
    assert len(out) == 16, f"expected 16 marker arms, got {len(out)}"
    return out


def ft_arms() -> list[Arm]:
    """The 16 #1586 full-FT arms (plan §4.1 amendment; identity pinned above)."""
    out = []
    for beh in FT_BEH_KEYS:
        for regime in REGIMES:
            for seed in SEEDS:
                arm_id = f"{beh}-pers-ft-{regime}-s{seed}"
                step, metric, _in_band = FT_SELECTED[arm_id]
                out.append(
                    Arm(
                        arm_id=arm_id,
                        kind="marker" if beh == "mk" else "content",
                        beh_key=beh,
                        ctx_key="pers",
                        regime=regime,
                        seed=seed,
                        lr=FT_LR,
                        step=step,
                        selection_read=metric,
                        method="ft",
                    )
                )
    assert len(out) == 16, len(out)
    return out


def all_arms(manifest: dict | None = None) -> list[Arm]:
    man = manifest or _load_manifest()
    arms = content_arms(man) + marker_arms(man) + ft_arms()
    assert len(arms) == 72, len(arms)  # 56 LoRA + 16 full-FT (plan §4.1)
    assert len({a.arm_id for a in arms}) == 72, "duplicate arm ids"
    return arms


def arm_method(arm_id: str) -> str:
    """'lora' | 'ft' for a registry arm id (base units are not arms)."""
    return {a.arm_id: a.method for a in all_arms()}[arm_id]


def reused_1586_arm(arm_id: str):
    """The #1586 ReusedLoraArm whose run_id == this verdict arm id (16 pers
    arms), or None. NOTE: LORA_ARM_BY_CELL keys are #1586 CELL names
    (`<beh>-pers-lora-<regime>-s<seed>`), not run ids — join on .run_id."""
    import issue1586_cells as g1586

    return next((a for a in g1586.REUSED_LORA_ARMS if a.run_id == arm_id), None)


def adapter_subfolder(arm: Arm) -> str:
    """Model-repo subfolder holding a LORA arm's selected checkpoint (§4.1).

    Resolution order: (1) issue1586_cells.REUSED_LORA_ARMS code-pinned
    subfolder (16 pers arms); (2) issue1481_cells reused-arm registries
    (fu4/fu5/fu7 con arms); (3) cas seed-42 reused-#1434 ladders; (4) fresh
    #1481 run convention (`issue1481[/marker]/<arm_id>/checkpoint-<step>`).
    A full-FT arm has NO adapter — resolve via `ft_ckpt_subfolder` instead.
    """
    if arm.method != "lora":
        raise ValueError(f"{arm.arm_id} is a {arm.method} arm — no adapter subfolder")
    reused = reused_1586_arm(arm.arm_id)
    if reused is not None:
        return reused.subfolder
    if arm.kind == "marker":
        return f"issue1481/marker/{arm.arm_id}/checkpoint-{arm.step}"
    import issue1481_cells as c1481

    rc = c1481.REUSED_CON_ARM_BY_ID.get(arm.arm_id)
    if rc is not None:
        return f"{rc.adapter_run_prefix}/checkpoint-{arm.step}"
    if arm.beh_key == "cas" and arm.seed == 42:
        lr_tag = arm.arm_id.split("-")[3]
        run_id = (
            f"ws-{arm.ctx_key}-{lr_tag}" if arm.regime == "con" else f"ws-po-{arm.ctx_key}-{lr_tag}"
        )
        return f"issue1434/{run_id}/checkpoint-{arm.step}"
    return f"issue1481/{arm.arm_id}/checkpoint-{arm.step}"


def ft_ckpt_subfolder(arm: Arm) -> str:
    """Overflow-repo path of a ft arm's selected FULL checkpoint (plan §4.1).

    Path symmetry with #1586's `_ckpt_persist_prefix` (`issue1586/<cell>/
    checkpoint-<step>`); the reused #1112 cell keeps its original overflow
    path. All 16 probed resolving on the Hub 2026-07-28; p0 re-probes.
    """
    assert arm.method == "ft", arm.arm_id
    if arm.arm_id == FT_REUSED_CELL:
        return FT_REUSED_SUBFOLDER
    return f"{FT_CKPT_HUB_PREFIX}/{arm.arm_id}/checkpoint-{arm.step}"


def delta_arm_for(arm: Arm) -> str:
    """The arm whose p5 δ cell this arm READS (plan §4.1 amendment).

    ft arms' δ cells COINCIDE with the 16 pers-LoRA cells — the #1586 ft
    cells trained on the SAME #1481 mixes at matched (beh, regime, seed) —
    so t̄_{C,B} is shared and p5 adds NO new cells; a LoRA arm owns its own.
    """
    if arm.method != "ft":
        return arm.arm_id
    import issue1586_cells as g1586

    return g1586.LORA_ARM_BY_CELL[f"{arm.beh_key}-pers-lora-{arm.regime}-s{arm.seed}"].run_id


def base_unit_for(arm_id: str) -> str:
    """Marker arms compare against the marker-decode base capture (plan §4.4)."""
    return "base_mk" if arm_id.startswith("mk-") else "base_content"


def max_new_tokens_for(unit_id: str) -> int:
    return MAX_NEW_MARKER if unit_id.startswith(("mk-", "base_mk")) else MAX_NEW_CONTENT


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


# ── corpus sample (plan §4.2) ────────────────────────────────────────────────


def pinned_split_block() -> dict:
    """The committed #779 n1M split block carrying the pinned val/test shas."""
    return json.loads(N1M_FITS_JSON.read_text())["split"]


def recover_valtest_prompts() -> list[str]:
    """Re-derive the 1,400 pinned val+test prompt strings (plan §4.2).

    Deterministic recovery the n1M driver documents: round1 = the first 5,000
    non-empty LMSYS first-turns (N50.sample_disjoint_n50k), then the ORIGINAL
    fixed_split(5000, 3600, 400, 1000, 42) val/test indices applied to it
    (N1G._valtest_prompts_from_round1, ctx0-guarded). Index-set shas are
    asserted against the committed pins by `assert_pinned_split()` callers.
    Bounded stream (~5k kept rows) — exempt from the checkpoint floor.
    """
    import issue779_ffc_n1m_generate_capture as n1g
    import issue779_ffc_n50k_generate_capture as n50g

    man = n50g.sample_disjoint_n50k(n1g.N_ROUND1, 0, 0)
    return n1g._valtest_prompts_from_round1(man["round1"])


def assert_pinned_split() -> dict:
    """Recompute fixed_split index shas and assert them against the pins."""
    import issue779_fitter_fair_comparison as f779

    pins = pinned_split_block()
    n_round1 = 5000
    _r1, val, test = f779.fixed_split(
        n_round1, n_round1 - N_VAL - N_TEST, N_VAL, N_TEST, f779.SPLIT_SEED
    )
    val_sha, test_sha = f779._sha_ids(val), f779._sha_ids(test)
    assert val_sha == pins["pinned_val_sha256"], (val_sha, pins["pinned_val_sha256"])
    assert test_sha == pins["pinned_test_sha256"], (test_sha, pins["pinned_test_sha256"])
    return {"val_sha256": val_sha, "test_sha256": test_sha, "n_val": N_VAL, "n_test": N_TEST}


def formatted_prompt_token_len(tokenizer, prompt: str) -> int:
    """Token length of the CHAT-TEMPLATED render (the #952 load-time rule)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def sample_train_prompts(
    pool: list[dict],
    meta: dict,
    tokenizer,
    valtest: list[str],
    *,
    n_train: int = N_TRAIN,
    seed: int = SAMPLE_SEED,
    token_cap: int = PROMPT_TOKEN_CAP,
) -> dict:
    """Deterministic stratified train sample from the n1M manifest pool.

    Stratified by corpus provenance at the manifest's REALIZED proportions
    (meta n_lmsys / n_wildchat), formatted-prompt token cap enforced at load
    (skip counts recorded), exact-text disjoint from the pinned val/test set,
    and sha-DEDUPED (#1768 crash-fix r4): the n1M pool contains exact-duplicate
    prompt texts (the #779 near-dupe screen was vs its 1,400 eval targets, never
    a within-corpus exact-dedup), so the seeded draw skips any candidate whose
    sha is already taken — within a corpus, across corpora, or vs the pinned
    val/test shas (val/test keep priority; train loses the row) — topping up
    from the continuing permutation order. Deterministic under ``seed``.
    Returns {rows, n_skipped_over_cap, n_skipped_valtest, n_skipped_dup,
    proportions}.
    """
    import numpy as np

    vt = set(valtest)
    taken: set[str] = {prompt_sha(p) for p in valtest}  # pinned rows keep priority
    by_corpus: dict[str, list[dict]] = {"lmsys": [], "wildchat": []}
    for r in pool:
        by_corpus[r["corpus"]].append(r)
    n_lm, n_wc = int(meta["n_lmsys"]), int(meta["n_wildchat"])
    quota_lm = round(n_train * n_lm / (n_lm + n_wc))
    quotas = {"lmsys": quota_lm, "wildchat": n_train - quota_lm}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    skipped_cap = 0
    skipped_vt = 0
    skipped_dup = 0
    for corpus, quota in quotas.items():
        cand = by_corpus[corpus]
        order = rng.permutation(len(cand))
        kept = 0
        for j in order:
            if kept >= quota:
                break
            p = cand[int(j)]["prompt"]
            if p in vt:
                skipped_vt += 1
                continue
            sha = prompt_sha(p)
            if sha in taken:  # duplicate text already drawn (or a pinned row)
                skipped_dup += 1
                continue
            if formatted_prompt_token_len(tokenizer, p) > token_cap:
                skipped_cap += 1
                continue
            rows.append({"prompt": p, "corpus": corpus, "sha": sha})
            taken.add(sha)
            kept += 1
        assert kept == quota, f"{corpus}: only {kept}/{quota} rows passed the caps"
    assert len(rows) == n_train, (len(rows), n_train)
    assert len({r["sha"] for r in rows}) == n_train, "train sha dedup failed (logic bug)"
    return {
        "rows": rows,
        "n_skipped_over_cap": skipped_cap,
        "n_skipped_valtest": skipped_vt,
        "n_skipped_dup": skipped_dup,
        "proportions": quotas,
        "seed": seed,
        "token_cap": token_cap,
    }


def load_corpus_sample(out_root: Path) -> dict:
    """The p0-built combined sample: ordered rows train+val+test with splits."""
    path = Path(out_root) / "inputs" / "corpus_sample.json"
    sample = json.loads(path.read_text())
    n = len(sample["rows"])
    assert n == sample["n_train"] + sample["n_val"] + sample["n_test"], n
    return sample
