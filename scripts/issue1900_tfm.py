"""#1900 same-issue follow-up `tfmargin-validation-expand` (plan v13) driver.

Expands the graded-DV external-validation leg from 1 arm x 299 contexts to
(12 content arms + base) x 3 family pools x 2 subsets. Three phases:

- ``prep``  (V0, VM):   freeze the fixed +/- pools per family (syc = #722 pools
  verbatim via ``build_fixed_pairs``; cas/imp = NEW pools from the parent's own
  judged completions), draw the scored context sets (800/subset/family, seeds
  19003/19004), commit + upload the config set to ``issue1900_leakrace/tfm/config/``.
- ``gpu``   (V1, pod):  ~22.7k teacher-forced margin units over 13 model states
  (1 base + 9 LoRA + 3 full-FT), sharded across every visible GPU (the parent
  ``issue1900_gpu`` worker fan-out, CVD pinned in the launcher env), pilot-gated
  per-unit basis with the pre-registered 800->500/subset descope lever;
  per-pass JSONL append + resume + per-pass HF upload before job end.
- ``stats`` (V2, VM):   30 Spearman reads rho(margin, graded) with B=2,000
  batched bootstrap CIs (reused ``issue1900_race`` rank machinery), the
  TWO-SAMPLE drift gate on the parent-replication cell, per-read tie fractions,
  the registered family-median verdict lattice, structure reads
  rho(margin, P1/P2/P7), and 3 figures.

Smoke (``--smoke``, plan section 4): 1 family (imp) x model states
{base, imp-pers-po-lr1e5-s42, imp-pers-ft-con-s42 (2 contexts — the full-FT
load-path class cell)} x 24 contexts, end-to-end prep -> gpu -> stats through
the SAME production entrypoints (smoke IS production at tiny N); smoke outputs
land under a disjoint ``tfm_smoke`` subtree + HF prefix (never the committed
canonical paths). ZERO new judge calls in every phase (graded scores are read
from the committed parent + off-floor artifacts).

Results reporting rides the dispatch handle/poller (fellows drained lane) —
no /workspace sentinel and no pod-side task.py, matching the parent driver.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1900_gpu as G  # noqa: E402  (ModelPool, fan-out helpers, MARGIN_CAP)
import issue1900_judge as J  # noqa: E402  (parent config loaders)
import issue1900_prep as P0  # noqa: E402  (HF prefix + corpus pin)
import issue1900_race as R  # noqa: E402  (bootstrap_battery / observed_rho)

logger = logging.getLogger("issue1900.tfm")

ISSUE = 1900
HF_PREFIX = P0.HF_PREFIX  # issue1900_leakrace
CONFIG_HF_PREFIX = P0.CONFIG_HF_PREFIX  # parent P0 config mirror (judge_filter et al.)
CORPUS_PIN = P0.CORPUS_PIN
PARQUET_PIN = "3bb20deb"  # parent predictor-table pin (issue1900_offfloor.py:81, verbatim)
FAMILIES = ("cas", "imp", "syc")
POOL_SIDE = G.MARGIN_CAP  # 32/side (plan section 0: MARGIN_CAP, issue1900_gpu.py:80)
POS_PER_ARM = 8  # plan section 4: top-8 per family arm
POS_SCORE_FLOOR = 50.0  # judged-positive floor (llm-judging rule 19 threshold)
POOL_PAIR_TOKEN_BUDGET = 768  # probe+answer token cap (plan section 4)
N_CTX_PER_SUBSET = 800  # plan section 4 scored context sets
DESCOPE_CTX_PER_SUBSET = 500  # pre-registered pilot descope lever (plan section 8)
SEED_PARENT_DRAW = 19003  # plan section 4
SEED_OFFFLOOR_DRAW = 19004  # plan section 4
BASIS_TFM_UNIT_S = 0.9  # plan section 9 p90 per-unit basis (parent P1c measured row)
PLANNED_V1_WALL_H = 1.6  # plan section 9 p90 wall
DRIFT_Z = 1.96  # plan section 6 line (1) two-sample gate
LATTICE_RHO_THRESHOLD = 0.15  # plan section 3 registered verdict threshold
READ_N_HARD_FLOOR = 4  # Spearman/CI math validity floor (any mode)
PROD_READ_N_FLOOR = 100  # production join-bug catch (descoped legit reads are >=~475)
B_BOOT = R.B_BOOT  # 2,000 (race CI convention)
TFM_HEADROOM_GB = 100.0  # plan section 9 mount-binding floor at the resolved out_root
SMOKE_FAMILY = "imp"
SMOKE_ARM = "imp-pers-po-lr1e5-s42"  # plan section 4 smoke LoRA state
SMOKE_FT_ARM = "imp-pers-ft-con-s42"  # full-FT load-path class cell (2 contexts)
SMOKE_CTX_PER_SUBSET = 12  # 12+12 -> <=24 contexts (plan section 4 smoke)
SMOKE_FT_CTX = 2
COMPOSITION = (
    "single user turn: f'{context}\\n\\n{probe}'; fixed pools identical across "
    "contexts (no selection-on-outcome)"
)
REPLICATION_ARM = "syc-pers-po-lr1e5-s42"  # parent P1c arm (drift-gate cell)


# ── configuration ────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    """Run configuration; smoke rebinds every OUTPUT surface to a tfm_smoke subtree."""

    out_root: Path
    stage_root: Path
    smoke: bool = False
    upload: bool = True
    worker_slot: int | None = None
    n_slots: int = 1

    @property
    def i1768_root(self) -> Path:
        return self.stage_root / X.HF_PREFIX

    @property
    def tfm_dir(self) -> Path:
        return self.out_root / ("tfm_smoke" if self.smoke else "tfm")

    @property
    def margins_dir(self) -> Path:
        return self.tfm_dir / "margins"

    @property
    def hf_tfm_prefix(self) -> str:
        return f"{HF_PREFIX}/tfm_smoke" if self.smoke else f"{HF_PREFIX}/tfm"

    @property
    def config_dir(self) -> Path:
        """V0 output config (committed canonical path in production; scratch in smoke)."""
        if self.smoke:
            return self.tfm_dir / "config"
        return REPO_ROOT / "eval_results/issue_1900/tfm/config"

    @property
    def stats_dir(self) -> Path:
        if self.smoke:
            return self.tfm_dir / "stats"
        return REPO_ROOT / "eval_results/issue_1900/tfm"

    @property
    def fig_dir(self) -> Path:
        if self.smoke:
            return self.tfm_dir / "figures"
        return REPO_ROOT / "figures/issue_1900"

    @property
    def parent_config_dir(self) -> Path:
        """Parent P0 config mirror (arms/subset/judge_filter); J loaders self-stage."""
        return REPO_ROOT / "data/issue_1900/config"

    @property
    def parent_judge_dir(self) -> Path:
        return REPO_ROOT / "eval_results/issue_1900/judge"

    @property
    def offfloor_judge_dir(self) -> Path:
        return REPO_ROOT / "eval_results/issue_1900/offfloor/judge"

    @property
    def offfloor_config_dir(self) -> Path:
        return REPO_ROOT / "eval_results/issue_1900/offfloor/config"

    def families(self) -> tuple[str, ...]:
        return (SMOKE_FAMILY,) if self.smoke else FAMILIES

    def gcfg(self) -> G.Cfg:
        """Parent-shaped Cfg for ModelPool.for_entry's staged-dir arithmetic."""
        return G.Cfg(out_root=self.out_root, stage_root=self.stage_root, smoke=False)


def load_arms(cfg: Cfg) -> list[dict]:
    """The 18-arm parent registry (local-first; J.load_arms stages from HF on a miss)."""
    return J.load_arms(cfg.parent_config_dir)


def family_arms(arms: list[dict], fam: str) -> list[dict]:
    """The family's 4 content arms in arms.json order (deterministic)."""
    out = [a for a in arms if a["kind"] == "content" and a["beh_key"] == fam]
    assert len(out) == 4, (fam, [a["arm_id"] for a in out])
    return out


def pools_path(cfg: Cfg, fam: str) -> Path:
    return cfg.config_dir / f"pools_{fam}.json"


def contexts_path(cfg: Cfg, fam: str) -> Path:
    return cfg.config_dir / f"contexts_{fam}.json"


def margins_paths(cfg: Cfg, fam: str, state: str) -> tuple[Path, Path]:
    """(jsonl, done) paths for one (family x model-state) margin pass."""
    stem = f"{fam}__{state}_margins"
    return cfg.margins_dir / f"{stem}.jsonl", cfg.margins_dir / f"{stem}.done.json"


def score_map(payload: dict) -> dict[str, float]:
    """sha -> graded score_mean over SCORED rows only (listwise; race convention)."""
    return {
        r["sha"]: float(r["score_mean"]) for r in payload["rows"] if r["score_mean"] is not None
    }


def _judge_payload(path: Path) -> dict:
    assert path.exists(), f"missing judge artifact: {path}"
    return json.loads(path.read_text())


# ── V0 prep: pool freeze + context draws ─────────────────────────────────────


def _tok_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _pair_within_budget(tok, probe: str, answer: str) -> bool:
    return _tok_len(tok, probe) + _tok_len(tok, answer) <= POOL_PAIR_TOKEN_BUDGET


def build_family_pools(
    fam: str,
    arm_ids: list[str],
    arm_payloads: dict[str, dict],
    base_payload: dict,
    raw_by_unit: dict[str, dict[str, dict]],
    prompt_by_sha: dict[str, str],
    tok,
) -> tuple[list[dict], list[dict], dict]:
    """Frozen +/- pools for cas/imp from the parent's own judged completions.

    Plan section 4: positives = top-8 per family arm by (score_mean desc, sha),
    score_mean >= 50, all 3 draws kept, one donor per distinct context ACROSS
    the pool, probe+answer <= 768 tokens (deterministic skip-and-take-next);
    negatives = 32 base rows with score_mean == 0 (all draws 0), sorted by sha,
    distinct contexts, disjoint from positive donors, same length filter —
    with the pre-registered lowest-score_mean fill when < 32 survive (section 8).
    Asserts the 100% donor-sha join (plan section 12 assumption 2) — a judged
    row missing from raw_rows/corpus_sample is a structural violation, never a
    skip. Returns (pos_pairs, neg_pairs, meta); pairs carry the exact
    {"probe", "answer"} keys ``compute_tf_margin`` consumes.
    """
    taken: set[str] = set()
    pos_pairs: list[dict] = []
    per_arm_depth: dict[str, dict] = {}

    def _pair(row_score: float, sha: str, unit: str) -> dict:
        raw = raw_by_unit[unit].get(sha)
        probe = prompt_by_sha.get(sha)
        assert raw is not None and probe is not None, (
            fam,
            unit,
            sha,
            "donor-sha join miss — raw_rows/corpus_sample must cover every judged row "
            "(plan section 12 assumption 2)",
        )
        return {
            "probe": probe,
            "answer": raw["response_text"],
            "score": float(row_score),
            "sha": sha,
            "source_arm": unit,
        }

    for arm_id in arm_ids:
        cands = [
            r
            for r in arm_payloads[arm_id]["rows"]
            if r["score_mean"] is not None
            and r["score_mean"] >= POS_SCORE_FLOOR
            and r["n_kept_draws"] == 3
        ]
        cands.sort(key=lambda r: (-r["score_mean"], r["sha"]))
        got, skipped_len, skipped_dup = 0, 0, 0
        for r in cands:
            if got == POS_PER_ARM:
                break
            if r["sha"] in taken:
                skipped_dup += 1
                continue
            pair = _pair(r["score_mean"], r["sha"], arm_id)
            if not _pair_within_budget(tok, pair["probe"], pair["answer"]):
                skipped_len += 1
                continue
            pos_pairs.append(pair)
            taken.add(r["sha"])
            got += 1
        per_arm_depth[arm_id] = {
            "n_candidates": len(cands),
            "n_taken": got,
            "skipped_over_budget": skipped_len,
            "skipped_duplicate_context": skipped_dup,
        }
        assert got == POS_PER_ARM, (
            fam,
            arm_id,
            got,
            f"positive pool depth below {POS_PER_ARM} after filters (plan section 12 "
            "assumption 1 — re-scope the pool rule before production)",
        )

    zero_rows = [
        r
        for r in base_payload["rows"]
        if r["score_mean"] is not None
        and r["score_mean"] == 0
        and all(s == 0 for s in r["kept_draw_scores"])
    ]
    zero_rows.sort(key=lambda r: r["sha"])
    neg_pairs: list[dict] = []
    n_zero_available = len(zero_rows)
    fallback_fill = 0

    def _try_neg(r: dict) -> bool:
        if r["sha"] in taken:
            return False
        pair = _pair(r["score_mean"], r["sha"], "base_content")
        if not _pair_within_budget(tok, pair["probe"], pair["answer"]):
            return False
        neg_pairs.append(pair)
        taken.add(r["sha"])
        return True

    for r in zero_rows:
        if len(neg_pairs) == POOL_SIDE:
            break
        _try_neg(r)
    if len(neg_pairs) < POOL_SIDE:
        # Pre-registered fallback (plan section 8): fill from the lowest-score_mean
        # scored base rows, same filters, recorded in the pool meta.
        rest = [
            r for r in base_payload["rows"] if r["score_mean"] is not None and r["score_mean"] > 0
        ]
        rest.sort(key=lambda r: (r["score_mean"], r["sha"]))
        for r in rest:
            if len(neg_pairs) == POOL_SIDE:
                break
            if _try_neg(r):
                fallback_fill += 1
    assert len(neg_pairs) == POOL_SIDE, (
        fam,
        len(neg_pairs),
        "negative pool below cap even after the lowest-score_mean fallback fill",
    )
    meta = {
        "n_pos_used": len(pos_pairs),
        "n_neg_used": len(neg_pairs),
        "cap": POOL_SIDE,
        "pos_rule": (
            f"top-{POS_PER_ARM} per family arm by (score_mean desc, sha); "
            f"score_mean >= {POS_SCORE_FLOOR}; all 3 draws kept; one donor per "
            "distinct context across the pool; deterministic skip-and-take-next"
        ),
        "neg_rule": "base rows with score_mean == 0 (all draws 0), sorted by sha",
        "neg_fallback_fill": fallback_fill,
        "n_zero_rows_available": n_zero_available,
        "pair_token_budget": POOL_PAIR_TOKEN_BUDGET,
        "per_arm_depth": per_arm_depth,
    }
    return pos_pairs, neg_pairs, meta


def _pool_content_sha(pos_pairs: list[dict], neg_pairs: list[dict]) -> str:
    blob = json.dumps({"pos": pos_pairs, "neg": neg_pairs}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_syc_pools(cfg: Cfg) -> tuple[list[dict], list[dict], dict]:
    """#722 sycophancy pools VERBATIM — byte-identical instrument to parent P1c."""
    from explore_persona_space.eval.margin import build_fixed_pairs
    from explore_persona_space.orchestrate import hub

    jf_path = cfg.parent_config_dir / "judge_filter.json"
    if not jf_path.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO, f"{CONFIG_HF_PREFIX}/judge_filter.json", jf_path, repo_type="dataset"
        )
    jf = json.loads(jf_path.read_text())
    pos_pairs, neg_pairs, pool_meta = build_fixed_pairs(jf, "sycophancy", cap=POOL_SIDE)
    meta = {
        **pool_meta,
        "provenance": (
            "#722 pools verbatim via build_fixed_pairs(judge_filter, 'sycophancy', "
            f"cap={POOL_SIDE}) — identical instrument to parent P1c (replication cell)"
        ),
        "donor_exclusion": "N/A — #661 extraction rollouts, disjoint from the #1768 corpus",
    }
    return pos_pairs, neg_pairs, meta


def _stage_raw_rows(cfg: Cfg, unit: str) -> None:
    """Per-file scoped staging of a unit's raw_rows shards @ CORPUS_PIN (#833)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dest = cfg.i1768_root / "corpus_capture" / unit
    if list(dest.glob("raw_rows_*.jsonl")):
        return
    listing = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            HfApi(),
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/corpus_capture/{unit}",
            repo_type="dataset",
            revision=CORPUS_PIN,
        ),
        what=f"raw_rows scoped listing ({unit})",
    )
    shards = sorted(p for p in listing if "raw_rows_" in Path(p).name and p.endswith(".jsonl"))
    assert shards, (unit, "no raw_rows shards at the pin")
    for p in shards:
        hub.stage_hub_file(
            X.HF_DATA_REPO, p, dest / Path(p).name, repo_type="dataset", revision=CORPUS_PIN
        )
    logger.info("[tfm-prep] %s: staged %d raw_rows shards", unit, len(shards))


def _stage_corpus_sample(cfg: Cfg) -> dict[str, str]:
    from explore_persona_space.orchestrate import hub

    local = cfg.i1768_root / "inputs" / "corpus_sample.json"
    if not local.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/inputs/corpus_sample.json",
            local,
            repo_type="dataset",
            revision=CORPUS_PIN,
        )
    sample = X.load_corpus_sample(cfg.i1768_root)
    return {r["sha"]: r["prompt"] for r in sample["rows"]}


def _draw(pool_sorted: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """Uniform without-replacement draw; DRAW ORDER preserved so a [:k] prefix
    is itself a uniform k-subdraw (the 800->500 descope lever, plan section 8)."""
    if len(pool_sorted) <= n:
        return list(pool_sorted)
    picked = rng.choice(np.asarray(pool_sorted, dtype=object), size=n, replace=False)
    return [str(s) for s in picked]


def draw_contexts(
    cfg: Cfg,
    fam: str,
    arm_ids: list[str],
    donors: list[str],
    prompt_by_sha: dict[str, str],
) -> dict:
    """Scored context sets (plan section 4): S_parent = 800 from the family's
    parent rows scored on ALL 4 arms AND base; S_offfloor = 800 from the frozen
    off-floor subset. Donors excluded from BOTH; 100% prompt coverage asserted."""
    donor_set = set(donors)
    scored_sets = [
        set(score_map(_judge_payload(cfg.parent_judge_dir / f"arm_scores_{a}.json")))
        for a in arm_ids
    ]
    scored_sets.append(
        set(score_map(_judge_payload(cfg.parent_judge_dir / f"arm_scores_base_{fam}.json")))
    )
    parent_pool = sorted(set.intersection(*scored_sets) - donor_set)
    off_subset = json.loads((cfg.offfloor_config_dir / f"subset_{fam}.json").read_text())
    assert off_subset["n"] == len(off_subset["shas"]), (fam, off_subset["n"])
    off_pool = sorted(set(off_subset["shas"]) - donor_set)
    for name, pool in (("parent", parent_pool), ("offfloor", off_pool)):
        missing = [s for s in pool if s not in prompt_by_sha]
        assert not missing, (
            fam,
            name,
            len(missing),
            "context shas missing from corpus_sample prompts (join violation)",
        )
    s_parent = _draw(parent_pool, N_CTX_PER_SUBSET, np.random.default_rng(SEED_PARENT_DRAW))
    s_off = _draw(off_pool, N_CTX_PER_SUBSET, np.random.default_rng(SEED_OFFFLOOR_DRAW))
    return {
        "family": fam,
        "S_parent": s_parent,
        "S_offfloor": s_off,
        "n_parent_pool": len(parent_pool),
        "n_offfloor_pool": len(off_pool),
        "donors_excluded": sorted(donor_set),
        "seeds": {"parent": SEED_PARENT_DRAW, "offfloor": SEED_OFFFLOOR_DRAW},
        "n_per_subset": N_CTX_PER_SUBSET,
    }


def phase_prep(cfg: Cfg) -> None:
    """V0 (VM): freeze pools + context draws; write + upload the tfm config set.

    Pool construction runs at FULL judged grain in BOTH modes (the smoke-slice
    probe of plan section 12 assumption 1 — a violated pool premise re-scopes
    the rule before any GPU dispatch); smoke restricts the FAMILY set only.
    """
    from transformers import AutoTokenizer

    G._phase("tfm_prep")
    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    done = cfg.config_dir / "prep_done.json"
    if done.exists():
        logger.info("[tfm-prep] done-sentinel present — skipping")
        return
    arms = load_arms(cfg)
    prompt_by_sha = _stage_corpus_sample(cfg)
    tok = AutoTokenizer.from_pretrained(X.BASE_MODEL)
    summary: dict[str, dict] = {}
    for fam in cfg.families():
        arm_ids = [a["arm_id"] for a in family_arms(arms, fam)]
        if fam == "syc":
            pos_pairs, neg_pairs, meta = build_syc_pools(cfg)
            donors: list[str] = []
        else:
            for unit in [*arm_ids, "base_content"]:
                _stage_raw_rows(cfg, unit)
            raw_by_unit = {u: G._read_raw_rows(cfg, u) for u in [*arm_ids, "base_content"]}
            arm_payloads = {
                a: _judge_payload(cfg.parent_judge_dir / f"arm_scores_{a}.json") for a in arm_ids
            }
            base_payload = _judge_payload(cfg.parent_judge_dir / f"arm_scores_base_{fam}.json")
            pos_pairs, neg_pairs, meta = build_family_pools(
                fam, arm_ids, arm_payloads, base_payload, raw_by_unit, prompt_by_sha, tok
            )
            donors = sorted({p["sha"] for p in pos_pairs} | {p["sha"] for p in neg_pairs})
        content_sha = _pool_content_sha(pos_pairs, neg_pairs)
        G._atomic_json(
            pools_path(cfg, fam),
            {
                "meta": G._meta(),
                "family": fam,
                "pos": pos_pairs,
                "neg": neg_pairs,
                "donors": donors,
                "pool_meta": meta,
                "content_sha256": content_sha,
            },
        )
        ctx = draw_contexts(cfg, fam, arm_ids, donors, prompt_by_sha)
        G._atomic_json(contexts_path(cfg, fam), {"meta": G._meta(), **ctx})
        summary[fam] = {
            "n_pos": len(pos_pairs),
            "n_neg": len(neg_pairs),
            "n_donors": len(donors),
            "content_sha256": content_sha,
            "n_parent_pool": ctx["n_parent_pool"],
            "n_offfloor_pool": ctx["n_offfloor_pool"],
        }
        print(
            f"[tfm-prep] {fam}: pools frozen (32+32, donors={len(donors)}, "
            f"sha={content_sha[:12]}) parent_pool={ctx['n_parent_pool']} "
            f"offfloor_pool={ctx['n_offfloor_pool']}",
            flush=True,
        )
    G._atomic_json(done, {"families": list(cfg.families()), "summary": summary, **G._meta()})
    if cfg.upload:
        from explore_persona_space.orchestrate import hub

        hub._upload(
            cfg.config_dir,
            X.HF_DATA_REPO,
            "dataset",
            f"{cfg.hf_tfm_prefix}/config",
            raise_on_error=True,
        )
        logger.info("[tfm-prep] config uploaded to %s/config", cfg.hf_tfm_prefix)
    print(f"[tfm-prep] done: {json.dumps(summary)}", flush=True)


# ── V1 gpu: margin passes ────────────────────────────────────────────────────


def ensure_tfm_config(cfg: Cfg) -> None:
    """Local-first -> HF-fetch -> fail-loud for the frozen pools/contexts set."""
    from explore_persona_space.orchestrate import hub

    for fam in cfg.families():
        for name in (f"pools_{fam}.json", f"contexts_{fam}.json"):
            local = cfg.config_dir / name
            if not local.exists():
                hub.stage_hub_file(
                    X.HF_DATA_REPO,
                    f"{cfg.hf_tfm_prefix}/config/{name}",
                    local,
                    repo_type="dataset",
                )
            assert local.exists(), f"tfm config unavailable locally or on HF: {name}"


def build_passes(cfg: Cfg, arms: list[dict]) -> list[dict]:
    """The (family x model-state) pass grid: 15 in production, 3 in smoke."""
    passes: list[dict] = []
    for fam in cfg.families():
        entries = family_arms(arms, fam)
        if cfg.smoke:
            entries = [a for a in entries if a["arm_id"] in (SMOKE_ARM, SMOKE_FT_ARM)]
            assert len(entries) == 2, [a["arm_id"] for a in entries]
        passes.append({"family": fam, "state": "base", "entry": None})
        for e in entries:
            p: dict = {"family": fam, "state": e["arm_id"], "entry": e}
            if cfg.smoke and e["arm_id"] == SMOKE_FT_ARM:
                p["ctx_limit"] = SMOKE_FT_CTX  # full-FT load-path class cell
            passes.append(p)
    expected = 3 if cfg.smoke else 15
    assert len(passes) == expected, (len(passes), expected)
    return passes


def pass_contexts(cfg: Cfg, fam: str) -> list[str]:
    """Union of the two subset draws, descope- and smoke-aware (sorted, resumable)."""
    ctx = json.loads(contexts_path(cfg, fam).read_text())
    if cfg.smoke:
        n = SMOKE_CTX_PER_SUBSET
    else:
        n = N_CTX_PER_SUBSET
        pilot = cfg.tfm_dir / "pilot.json"
        if pilot.exists():
            n = int(json.loads(pilot.read_text())["n_ctx_per_subset"].get(fam, n))
    return sorted(set(ctx["S_parent"][:n]) | set(ctx["S_offfloor"][:n]))


def run_tfm_pass(
    cfg: Cfg,
    pool: G.ModelPool,
    p: dict,
    prompt_by_sha: dict[str, str],
    limit: int | None = None,
) -> list[float]:
    """One (family x model-state) margin pass — the parent run_p1c_side pattern.

    Per-context JSONL append + resume (checkpoint-per-unit; >50-unit trigger),
    `[tfm] unit k/N` progress lines, done-sentinel + per-pass HF upload on
    completion. ``limit`` slices the PENDING contexts (pilot mode); the done
    sentinel is only written on a full (limit=None) pass. Returns per-unit
    wall seconds for the units this call computed.
    """
    from explore_persona_space.eval.margin import compute_tf_margin

    fam, state = p["family"], p["state"]
    out, done = margins_paths(cfg, fam, state)
    out.parent.mkdir(parents=True, exist_ok=True)
    if done.exists():
        return []
    pools = json.loads(pools_path(cfg, fam).read_text())
    pos_pairs, neg_pairs = pools["pos"], pools["neg"]
    assert len(pos_pairs) == POOL_SIDE and len(neg_pairs) == POOL_SIDE, (
        fam,
        len(pos_pairs),
        len(neg_pairs),
    )
    shas = pass_contexts(cfg, fam)
    if p.get("ctx_limit"):
        shas = shas[: p["ctx_limit"]]
    done_shas: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as fh:  # text-mode iteration (U+2028 trap)
            for line in fh:
                if line.strip():
                    done_shas.add(json.loads(line)["sha"])
    pending = [s for s in shas if s not in done_shas]
    if limit is not None:
        pending = pending[:limit]
    tok = pool.tokenizer()
    device = next(pool.base().parameters()).device
    unit_times: list[float] = []
    t0 = time.time()
    with pool.for_entry(cfg.gcfg(), p["entry"]) as model:
        for k, sha in enumerate(pending):
            ctx_text = prompt_by_sha[sha]

            def messages_fn(probe: str, _ctx: str = ctx_text) -> list[dict]:
                return [{"role": "user", "content": f"{_ctx}\n\n{probe}"}]

            t1 = time.time()
            res = compute_tf_margin(model, tok, messages_fn, pos_pairs, neg_pairs, device=device)
            unit_times.append(time.time() - t1)
            rec = {
                "sha": sha,
                "family": fam,
                "state": state,
                "margin": res.margin,
                "pos_mean_ln_logp": res.pos_mean_ln_logp,
                "neg_mean_ln_logp": res.neg_mean_ln_logp,
                "n_pos": res.n_pos,
                "n_neg": res.n_neg,
            }
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"[tfm] unit {k + 1}/{len(pending)} {fam}:{state}:{sha[:8]} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    if limit is not None:
        return unit_times
    G._atomic_json(
        done,
        {
            "family": fam,
            "state": state,
            "n_contexts": len(shas),
            "cap_per_side": POOL_SIDE,
            "pool_content_sha256": pools["content_sha256"],
            "composition": COMPOSITION,
            "unit_s_mean": round(float(np.mean(unit_times)), 3) if unit_times else None,
            **G._meta(),
        },
    )
    if cfg.upload:
        _upload_pass(cfg, out, done)
    return unit_times


def _upload_pass(cfg: Cfg, out: Path, done: Path) -> None:
    """Per-pass margin upload (text/JSON — unconditional per the upload policy)."""
    from explore_persona_space.orchestrate import hub

    prefix = f"{cfg.hf_tfm_prefix}/margins"
    hub._upload(
        out,
        X.HF_DATA_REPO,
        "dataset",
        f"{prefix}/{out.name}",
        upload_as_file=True,
        raise_on_error=True,
    )
    hub._upload(
        done,
        X.HF_DATA_REPO,
        "dataset",
        f"{prefix}/{done.name}",
        upload_as_file=True,
        raise_on_error=True,
    )


def _prompt_map(cfg: Cfg) -> dict[str, str]:
    sample = X.load_corpus_sample(cfg.i1768_root)
    return {r["sha"]: r["prompt"] for r in sample["rows"]}


def stage_inputs(cfg: Cfg, arms: list[dict]) -> None:
    """V1 pod staging: corpus sample @ pin + the pass-needed model artifacts.

    LoRA adapters and full-FT checkpoints stage via the verbatim prefix mirror
    (files land at dest/<repo-relative path>); the consumer-open probe asserts
    the consumer-side dir per staged arm BEFORE any model load (artifact-reuse
    (h)(iv); #928/#1481 class). The 3 content-FT checkpoints were never
    model-loaded by the parent (banked-store consumers), so they stage HERE.
    """
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    free = assert_out_root_headroom(cfg.out_root, TFM_HEADROOM_GB, phase="tfm_stage")
    logger.info("[tfm-stage] headroom OK: %.1f GB free at %s", free, cfg.out_root)
    _stage_corpus_sample(cfg)
    done = cfg.tfm_dir / "stage.done.json"
    if done.exists():
        logger.info("[tfm-stage] done-sentinel present — skipping model staging")
        return
    entries = [p["entry"] for p in build_passes(cfg, arms) if p["entry"] is not None]
    for a in entries:
        if a["method"] == "lora":
            dest = cfg.stage_root / "adapters" / a["arm_id"]
            hub.stage_hub_prefix(a["adapter_repo"], a["adapter_subfolder"], dest, repo_type="model")
            probe = G._adapter_dir(cfg.gcfg(), a) / "adapter_config.json"
        else:
            dest = cfg.stage_root / "ft_ckpt" / a["arm_id"]
            hub.stage_hub_prefix(a["ft_repo"], a["ft_subfolder"], dest, repo_type="model")
            probe = G._ft_dir(cfg.gcfg(), a) / "config.json"
        assert probe.exists(), (
            f"staged-layout consumer-open FAIL for {a['arm_id']}: {probe} missing after "
            "staging ((h)(iv); #928/#1481)"
        )
    G._atomic_json(done, {"n_model_states": len(entries), **G._meta()})
    logger.info("[tfm-stage] complete: %d model states staged", len(entries))


def gpu_pilot(cfg: Cfg, arms: list[dict]) -> None:
    """Pilot-gate (plan sections 8/12 assumption 3): time first units per family
    through the production pass entrypoint; >=2x basis -> recorded descope to
    500/subset. Two units are timed and the MIN taken (warmup-robust; biased
    AGAINST a spurious descope). Runs in a CVD-pinned subprocess so its HBM is
    freed before the worker fan-out."""
    prompt_by_sha = _prompt_map(cfg)
    pool = G.ModelPool(G._device(), G._dtype())
    per_family: dict[str, dict] = {}
    n_ctx: dict[str, int] = {}
    for fam in cfg.families():
        p = {"family": fam, "state": "base", "entry": None}
        times = run_tfm_pass(cfg, pool, p, prompt_by_sha, limit=2)
        unit_s = min(times) if times else None  # None: pass already complete (resume)
        descope = unit_s is not None and unit_s >= G.DEVIATION_MULT * BASIS_TFM_UNIT_S
        per_family[fam] = {"unit_s": times, "unit_s_min": unit_s, "descope": descope}
        n_ctx[fam] = DESCOPE_CTX_PER_SUBSET if descope else N_CTX_PER_SUBSET
        if descope:
            print(
                f"[tfm-pilot] family={fam} unit_s_min={unit_s:.2f} >= "
                f"{G.DEVIATION_MULT}x basis {BASIS_TFM_UNIT_S}s — descope "
                f"{N_CTX_PER_SUBSET}->{DESCOPE_CTX_PER_SUBSET}/subset (plan section 8 lever)",
                flush=True,
            )
    G._atomic_json(
        cfg.tfm_dir / "pilot.json",
        {
            "per_family": per_family,
            "basis_s": BASIS_TFM_UNIT_S,
            "n_ctx_per_subset": n_ctx,
            **G._meta(),
        },
    )
    print(f"[tfm-pilot] done: {json.dumps(n_ctx)}", flush=True)


def _pilot_deviation_row(cfg: Cfg, n_gpus: int) -> dict | None:
    """Post-pilot projection vs the plan section 9 row (poller-visible line)."""
    pilot_path = cfg.tfm_dir / "pilot.json"
    if not pilot_path.exists():
        return None
    pilot = json.loads(pilot_path.read_text())
    unit_ss = [v["unit_s_min"] for v in pilot["per_family"].values() if v["unit_s_min"] is not None]
    if not unit_ss:
        return None
    worst = max(unit_ss)
    n_units = sum(
        len(pass_contexts(cfg, fam)) * 5 for fam in cfg.families()
    )  # 5 model states/family
    projected_h = n_units * worst / max(n_gpus, 1) / 3600.0
    return G._deviation(
        "V1 tfm margins",
        PLANNED_V1_WALL_H,
        projected_h,
        f"pilot first-unit min {worst:.2f}s x {n_units} units / {n_gpus} GPUs "
        f"(plan basis {BASIS_TFM_UNIT_S}s p90)",
    )


def _spawn_tfm_workers(cfg: Cfg, argv_tail: list[str]) -> None:
    """Fan the pass list across every visible GPU (CVD pinned in the LAUNCHER env)."""
    gpu_ids = G._physical_gpu_ids()
    n = len(gpu_ids)
    logdir = cfg.tfm_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    procs = []
    for slot, gid in enumerate(gpu_ids):
        log = logdir / f"worker{slot}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "gpu",
            "--worker-slot",
            str(slot),
            "--n-slots",
            str(n),
            *argv_tail,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gid}
        fh = log.open("a")
        procs.append((slot, log, fh, subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)))
        logger.info("[tfm-fanout] worker %d -> GPU %s (pid %d)", slot, gid, procs[-1][3].pid)
    failed = []
    for slot, log, fh, proc in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            failed.append((slot, rc))
            # JSONL_SPLITLINES_EXEMPT: worker .log tail read (free-text log lines)
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
            print(f"[tfm-fanout] worker {slot} FAILED rc={rc}; log tail:", flush=True)
            print("\n".join(tail), flush=True)
    assert not failed, f"tfm worker subprocesses failed: {failed}"
    done = sorted(logdir.glob("worker*.done.json"))
    assert len(done) == n, (len(done), n, "missing tfm worker done-sentinels")


def gpu_worker(cfg: Cfg, arms: list[dict]) -> None:
    """Per-GPU worker: run passes where idx % n_slots == worker_slot."""
    prompt_by_sha = _prompt_map(cfg)
    passes = build_passes(cfg, arms)
    mine = [(i, p) for i, p in enumerate(passes) if i % cfg.n_slots == cfg.worker_slot]
    pool = G.ModelPool(G._device(), G._dtype())
    for k, (idx, p) in enumerate(mine):
        t0 = time.time()
        run_tfm_pass(cfg, pool, p, prompt_by_sha)
        print(
            f"[tfm-worker{cfg.worker_slot}] pass {k + 1}/{len(mine)} item#{idx} "
            f"{p['family']}:{p['state']} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    G._atomic_json(
        cfg.tfm_dir / "logs" / f"worker{cfg.worker_slot}.done.json",
        {"slot": cfg.worker_slot, "n_passes": len(mine), **G._meta()},
    )


def phase_gpu(cfg: Cfg, pilot_only: bool = False) -> None:
    """V1: stage -> pilot (production) -> per-GPU pass fan-out -> upload verify."""
    G._phase("tfm_gpu")
    arms = load_arms(cfg)
    ensure_tfm_config(cfg)
    (cfg.tfm_dir / "logs").mkdir(parents=True, exist_ok=True)
    if cfg.worker_slot is not None:
        gpu_worker(cfg, arms)
        return
    if pilot_only:
        gpu_pilot(cfg, arms)
        return
    G._device()  # fail loud with no CUDA (plan section 9 preamble)
    stage_inputs(cfg, arms)
    argv_tail = [
        "--out-root",
        str(cfg.out_root),
        "--stage-root",
        str(cfg.stage_root),
    ]
    if cfg.smoke:
        argv_tail.append("--smoke")
    if not cfg.upload:
        argv_tail.append("--no-upload")
    gpu_ids = G._physical_gpu_ids()
    deviation = None
    if not cfg.smoke:
        _run_subprocess_leg(
            cfg,
            ["gpu", "--pilot-only", *argv_tail],
            log_name="pilot.log",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids[0]},
        )
        assert (cfg.tfm_dir / "pilot.json").exists(), "pilot subprocess wrote no pilot.json"
        deviation = _pilot_deviation_row(cfg, len(gpu_ids))
    _spawn_tfm_workers(cfg, argv_tail)
    passes = build_passes(cfg, arms)
    missing = [
        f"{p['family']}__{p['state']}"
        for p in passes
        if not margins_paths(cfg, p["family"], p["state"])[1].exists()
    ]
    assert not missing, f"passes without done-sentinels after fan-out: {missing}"
    all_done = cfg.margins_dir / "all_done.json"
    G._atomic_json(
        all_done,
        {
            "n_passes": len(passes),
            "passes": [f"{p['family']}__{p['state']}" for p in passes],
            "pilot_deviation": deviation,
            "smoke": cfg.smoke,
            **G._meta(),
        },
    )
    if cfg.upload:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        hub._upload(
            all_done,
            X.HF_DATA_REPO,
            "dataset",
            f"{cfg.hf_tfm_prefix}/margins/all_done.json",
            upload_as_file=True,
            raise_on_error=True,
        )
        pilot_path = cfg.tfm_dir / "pilot.json"
        if pilot_path.exists():
            hub._upload(
                pilot_path,
                X.HF_DATA_REPO,
                "dataset",
                f"{cfg.hf_tfm_prefix}/margins/pilot.json",
                upload_as_file=True,
                raise_on_error=True,
            )
        prefix = f"{cfg.hf_tfm_prefix}/margins"
        expected = [f"{prefix}/all_done.json"]
        for p in passes:
            out, done = margins_paths(cfg, p["family"], p["state"])
            expected += [f"{prefix}/{out.name}", f"{prefix}/{done.name}"]
        missing_hub = hub.verify_repo_paths_uploaded(
            HfApi(), X.HF_DATA_REPO, expected, path_in_repo=prefix, repo_type="dataset"
        )
        assert not missing_hub, f"margin uploads missing on the Hub: {missing_hub}"
        logger.info("[tfm-gpu] upload verify PASS (%d files)", len(expected))
    print(f"[tfm-gpu] done: {len(passes)} passes complete", flush=True)


def _run_subprocess_leg(cfg: Cfg, argv: list[str], *, log_name: str, env: dict) -> None:
    """Run one driver subprocess leg (pilot) with an explicit env + tail-on-fail."""
    log = cfg.tfm_dir / "logs" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), *argv]
    with log.open("a") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=fh, env=env).returncode
    if rc != 0:
        # JSONL_SPLITLINES_EXEMPT: subprocess .log tail read (free-text log lines)
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
        print(f"[tfm] subprocess leg {argv[:2]} FAILED rc={rc}; log tail:", flush=True)
        print("\n".join(tail), flush=True)
        raise RuntimeError(f"tfm subprocess leg failed rc={rc}: {argv[:2]}")


# ── V2 stats: validation reads + drift gate + lattice + figures ──────────────


def _read_margin_rows(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    with path.open(encoding="utf-8") as fh:  # text-mode iteration (U+2028 trap)
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["sha"]] = float(r["margin"])
    return rows


def load_margins(cfg: Cfg, passes: list[dict]) -> dict[tuple[str, str], dict[str, float]]:
    """Local-first -> HF-fetch -> fail-loud (production) / skip+log (smoke)."""
    from explore_persona_space.orchestrate import hub

    out: dict[tuple[str, str], dict[str, float]] = {}
    for p in passes:
        fam, state = p["family"], p["state"]
        path, _done = margins_paths(cfg, fam, state)
        if not path.exists():
            try:
                hub.stage_hub_file(
                    X.HF_DATA_REPO,
                    f"{cfg.hf_tfm_prefix}/margins/{path.name}",
                    path,
                    repo_type="dataset",
                )
            except Exception as exc:  # noqa: BLE001 — mode-split below keeps fail-loud
                if not cfg.smoke:
                    raise AssertionError(
                        f"margin pass unavailable locally or on HF: {path.name} ({exc})"
                    ) from exc
                logger.warning("[tfm-stats] SMOKE: pass %s missing — skipped", path.name)
                continue
        out[(fam, state)] = _read_margin_rows(path)
    return out


def graded_map(cfg: Cfg, fam: str, state: str, subset: str) -> dict[str, float]:
    """Graded score_mean joins per (state, subset) — plan section 3 row-coverage."""
    name = f"base_{fam}" if state == "base" else state
    d = cfg.parent_judge_dir if subset == "parent" else cfg.offfloor_judge_dir
    return score_map(_judge_payload(d / f"arm_scores_{name}.json"))


def _read_seed(fam: str, state: str, subset: str) -> int:
    return 19005 + (zlib.crc32(f"{fam}:{state}:{subset}".encode()) % 100_000)


def _tie_stats(vals: np.ndarray) -> dict:
    _uniq, counts = np.unique(vals, return_counts=True)
    return {
        "tie_frac_modal": float(counts.max() / len(vals)),
        "frac_zero": float(np.mean(vals == 0.0)),
    }


def _rho_and_ci(x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Observed Spearman + B=2,000 batched bootstrap CI (race rank machinery)."""
    xm = x.reshape(-1, 1).astype(np.float64)
    ym = y.reshape(-1, 1).astype(np.float64)
    rho = float(R.observed_rho(xm, ym)[0, 0])
    draws, n_degen = R.bootstrap_battery(xm, ym, B_BOOT, seed=seed)
    d = draws[:, 0, 0].astype(np.float64)
    lo, hi = (float(q) for q in np.quantile(d, R.CI_QS))
    return {
        "rho": rho,
        "ci_lo": lo,
        "ci_hi": hi,
        "se_boot": float(d.std(ddof=1)),
        "b_draws": B_BOOT,
        "n_degenerate_series_draws": int(n_degen),
    }


def stats_reads(
    cfg: Cfg,
    arms: list[dict],
    margins: dict[tuple[str, str], dict[str, float]],
) -> list[dict]:
    """The 30 registered validation reads: (12 arms + 3 base) x 2 subsets."""
    reads: list[dict] = []
    for fam in cfg.families():
        ctx = json.loads(contexts_path(cfg, fam).read_text())
        entries = family_arms(arms, fam)
        if cfg.smoke:
            entries = [a for a in entries if a["arm_id"] in (SMOKE_ARM, SMOKE_FT_ARM)]
        states = ["base"] + [a["arm_id"] for a in entries]
        for state in states:
            m = margins.get((fam, state))
            if m is None:
                assert cfg.smoke, (fam, state, "missing margin pass in production")
                continue
            for subset, shas in (("parent", ctx["S_parent"]), ("offfloor", ctx["S_offfloor"])):
                g = graded_map(cfg, fam, state, subset)
                pairs = [(m[s], g[s]) for s in shas if s in m and s in g]
                n = len(pairs)
                if n < READ_N_HARD_FLOOR:
                    assert cfg.smoke, (fam, state, subset, n, "read below hard floor")
                    logger.warning(
                        "[tfm-stats] SMOKE: read %s:%s:%s n=%d < %d — skipped",
                        fam,
                        state,
                        subset,
                        n,
                        READ_N_HARD_FLOOR,
                    )
                    continue
                if not cfg.smoke and n < PROD_READ_N_FLOOR:
                    raise AssertionError(
                        (fam, state, subset, n, "production read implausibly small — join bug?")
                    )
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])
                read = {
                    "family": fam,
                    "state": state,
                    "subset": subset,
                    "grain": "per-prompt (NOT #722's cell grain)",
                    "n": n,
                    "n_margin_rows": len(m),
                    "n_subset_draw": len(shas),
                    **_rho_and_ci(x, y, _read_seed(fam, state, subset)),
                    **_tie_stats(y),
                }
                reads.append(read)
    return reads


def drift_flag(
    rho_new: float, rho_parent: float, se_parent: float, se_new: float
) -> tuple[bool, float]:
    """Two-sample machinery-drift gate (plan section 6 line 1).

    Returns (flagged, threshold) with threshold = 1.96 * sqrt(SE_p^2 + SE_n^2);
    flagged iff |rho_new - rho_parent| > threshold.
    """
    threshold = DRIFT_Z * math.sqrt(se_parent**2 + se_new**2)
    return abs(rho_new - rho_parent) > threshold, threshold


def _parent_se(cfg: Cfg) -> tuple[float, str, int]:
    """SE_parent via bootstrap re-reduction of the persisted parent per-context
    frame (HF validation/tf_margin_arm.jsonl x parent judge scores); registered
    fallback: analytic 1/sqrt(n-3) at the parent's persisted n (plan section 6)."""
    parent = json.loads(
        (REPO_ROOT / "eval_results/issue_1900/race/validation_read.json").read_text()
    )
    n_parent = int(parent["n_overlap_contexts"])
    try:
        from explore_persona_space.orchestrate import hub

        local = cfg.tfm_dir / "parent_frame" / "tf_margin_arm.jsonl"
        if not local.exists():
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{HF_PREFIX}/validation/tf_margin_arm.jsonl",
                local,
                repo_type="dataset",
            )
        m = _read_margin_rows(local)
        g = graded_map(cfg, "syc", REPLICATION_ARM, "parent")
        pairs = [(m[s], g[s]) for s in m if s in g]
        assert len(pairs) >= READ_N_HARD_FLOOR, len(pairs)
        x = np.array([p[0] for p in pairs]).reshape(-1, 1)
        y = np.array([p[1] for p in pairs]).reshape(-1, 1)
        draws, _ = R.bootstrap_battery(x, y, B_BOOT, seed=19001)
        se = float(draws[:, 0, 0].astype(np.float64).std(ddof=1))
        return se, f"bootstrap re-reduction of the parent frame (n={len(pairs)})", n_parent
    except Exception as exc:  # noqa: BLE001 — plan-registered analytic fallback, recorded
        se = 1.0 / math.sqrt(n_parent - 3)
        logger.warning("[tfm-stats] parent-frame bootstrap unavailable (%s) — analytic SE", exc)
        return se, f"analytic 1/sqrt(n-3) fallback (n={n_parent}; {type(exc).__name__})", n_parent


def drift_gate(cfg: Cfg, reads: list[dict]) -> dict:
    """Parent-replication cell vs the persisted -0.064 (blocks interpretation on flag)."""
    rep = next(
        (r for r in reads if r["state"] == REPLICATION_ARM and r["subset"] == "parent"), None
    )
    if rep is None:
        assert cfg.smoke, "replication cell missing in production stats"
        return {"status": "skipped", "reason": "smoke — replication cell not in the smoke grid"}
    parent = json.loads(
        (REPO_ROOT / "eval_results/issue_1900/race/validation_read.json").read_text()
    )
    rho_parent = float(parent["rho_margin_graded"])
    se_parent, se_src, n_parent = _parent_se(cfg)
    flagged, threshold = drift_flag(rep["rho"], rho_parent, se_parent, rep["se_boot"])
    return {
        "status": "flagged" if flagged else "ok",
        "arm_id": REPLICATION_ARM,
        "rho_new": rep["rho"],
        "n_new": rep["n"],
        "rho_parent": rho_parent,
        "n_parent": n_parent,
        "se_parent": se_parent,
        "se_parent_source": se_src,
        "se_new": rep["se_boot"],
        "threshold": threshold,
        "rule": "flag iff |rho_new - rho_parent| > 1.96*sqrt(SE_parent^2 + SE_new^2)",
        "note": "a flagged drift blocks interpretation of all 30 reads (plan section 6)",
    }


def lattice_verdict(reads: list[dict], smoke: bool) -> dict:
    """Registered DISJOINT+exhaustive verdict lattice (plan section 3):
    family medians of the OFF-FLOOR arm reads vs the 0.15 threshold."""
    medians: dict[str, float] = {}
    qualifications: dict[str, list[str]] = {}
    for fam in FAMILIES:
        arm_reads = [
            r
            for r in reads
            if r["family"] == fam and r["subset"] == "offfloor" and r["state"] != "base"
        ]
        if len(arm_reads) < 4:
            if smoke:
                return {
                    "status": "partial",
                    "reason": f"smoke — {fam} has {len(arm_reads)}/4 off-floor arm reads",
                }
            raise AssertionError((fam, len(arm_reads), "lattice needs 4 off-floor arm reads"))
        medians[fam] = float(np.median([r["rho"] for r in arm_reads]))
        qualifications[fam] = [
            r["state"] for r in arm_reads if r["ci_lo"] <= LATTICE_RHO_THRESHOLD <= r["ci_hi"]
        ]
    mn, mx = min(medians.values()), max(medians.values())
    if mn > LATTICE_RHO_THRESHOLD:
        verdict = "uniformly-validates"
    elif mx <= LATTICE_RHO_THRESHOLD:
        verdict = "uniformly-fails"
    else:
        verdict = "behavior-specific"
    return {
        "status": "ok",
        "verdict": verdict,
        "threshold": LATTICE_RHO_THRESHOLD,
        "family_medians_offfloor": medians,
        "arms_with_ci_spanning_threshold": qualifications,
        "note": (
            "verdict read on point-estimate medians; a median straddling the threshold "
            "with CIs spanning it is narrated as unresolved-at-this-n (plan section 6 line 6)"
        ),
    }


def _stage_parent_parquet(cfg: Cfg, arm_id: str, layer: int) -> Path:
    from explore_persona_space.orchestrate import hub

    dest = cfg.tfm_dir / "parent_tables" / f"{arm_id}_L{layer}.parquet"
    if not dest.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{HF_PREFIX}/predictor_tables/{arm_id}_L{layer}.parquet",
            dest,
            repo_type="dataset",
            revision=PARQUET_PIN,
        )
    return dest


def structure_reads(
    cfg: Cfg,
    arms: list[dict],
    margins: dict[tuple[str, str], dict[str, float]],
) -> list[dict]:
    """rho(margin, P1/P2/P7) per arm on the parent subset (plan section 6 line 4;
    parent validation_read precedent) — discriminates 'margin is noise' from
    'margin tracks geometry but not the judge'. Pure re-reductions."""
    import pandas as pd

    out: list[dict] = []
    for fam in cfg.families():
        ctx = json.loads(contexts_path(cfg, fam).read_text())
        parent_shas = set(ctx["S_parent"])
        base_scores = graded_map(cfg, fam, "base", "parent")  # p7 = base propensity
        entries = family_arms(arms, fam)
        if cfg.smoke:
            entries = [a for a in entries if a["arm_id"] in (SMOKE_ARM, SMOKE_FT_ARM)]
        for a in entries:
            m = margins.get((fam, a["arm_id"]))
            if m is None:
                assert cfg.smoke, (fam, a["arm_id"], "missing margin pass")
                continue
            layer = int(a["primary_layer"])
            tab = pd.read_parquet(_stage_parent_parquet(cfg, a["arm_id"], layer))
            tab = tab[tab["in_judge_subset"]].reset_index(drop=True)
            tab = tab[tab["sha"].isin(parent_shas)].reset_index(drop=True)
            mv = tab["sha"].map(m).to_numpy(float)
            p7 = tab["sha"].map(base_scores).to_numpy(float)
            rec = {"family": fam, "arm_id": a["arm_id"], "layer": layer}
            for cand, col in (("p1", "p1_tc"), ("p2", "p2_tc")):
                cv = tab[col].to_numpy(float)
                ok = ~np.isnan(mv) & ~np.isnan(cv)
                rec[f"rho_margin_{cand}"] = (
                    R._spearman_np(mv[ok], cv[ok]) if int(ok.sum()) >= 10 else None
                )
                rec[f"n_{cand}"] = int(ok.sum())
            ok = ~np.isnan(mv) & ~np.isnan(p7)
            rec["rho_margin_p7"] = R._spearman_np(mv[ok], p7[ok]) if int(ok.sum()) >= 10 else None
            rec["n_p7"] = int(ok.sum())
            out.append(rec)
    return out


def _figures(cfg: Cfg, reads: list[dict], drift: dict, margins, arms) -> list[Path]:
    """3 figures, plain-English labels (no bare P#/M#/arm codes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import issue1900_figs as FG

    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    subset_names = {"parent": "parent judge subset", "offfloor": "off-floor subset"}
    subset_colors = {"parent": "#888888", "offfloor": "#0072B2"}

    def state_plain(state: str, fam: str) -> str:
        if state == "base":
            return f"base model ({FG.BEH_NAMES.get(fam, fam)} pool)"
        return FG.arm_plain(state)

    # (1) forest plot: every read's rho + bootstrap CI, grouped by family.
    rows = sorted(reads, key=lambda r: (r["family"], r["state"] == "base", r["state"]))
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(rows) + 1.5)), layout="constrained")
    ys, labels = [], []
    for i, r in enumerate(rows):
        y = len(rows) - i
        off = 0.18 if r["subset"] == "offfloor" else -0.18
        # non-negative offsets from the point value (matplotlib xerr contract)
        lo = max(0.0, r["rho"] - r["ci_lo"])
        hi = max(0.0, r["ci_hi"] - r["rho"])
        ax.errorbar(
            r["rho"],
            y + off,
            xerr=[[lo], [hi]],
            fmt="o",
            markersize=3.5,
            color=subset_colors[r["subset"]],
            elinewidth=1.0,
        )
        if r["subset"] == "parent":
            ys.append(y)
            labels.append(state_plain(r["state"], r["family"]))
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(LATTICE_RHO_THRESHOLD, color="#D55E00", linewidth=0.9, linestyle="--")
    ax.set_yticks(ys, labels, fontsize=7)
    ax.set_xlabel("Spearman rho(TF margin, graded judge score) per prompt")
    handles = [
        plt.Line2D([], [], color=c, marker="o", linestyle="", label=subset_names[s])
        for s, c in subset_colors.items()
    ]
    handles.append(
        plt.Line2D([], [], color="#D55E00", linestyle="--", label="0.15 verdict threshold")
    )
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    ax.set_title("TF-margin external validation — 95% bootstrap CIs (B=2,000)", fontsize=9)
    made.append(FG._save(fig, "tfm_validation_forest", cfg.fig_dir))

    # (2) margin-vs-graded scatter (off-floor subset), one panel per arm state.
    panel_keys = [(r["family"], r["state"]) for r in rows if r["subset"] == "offfloor"]
    panel_keys = list(dict.fromkeys(panel_keys))
    ncol = 5 if len(panel_keys) > 5 else max(1, len(panel_keys))
    nrow = int(np.ceil(len(panel_keys) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(2.9 * ncol, 2.6 * nrow), layout="constrained", squeeze=False
    )
    for i, (fam, state) in enumerate(panel_keys):
        ax = axes[i // ncol][i % ncol]
        ctx = json.loads(contexts_path(cfg, fam).read_text())
        m = margins.get((fam, state), {})
        g = graded_map(cfg, fam, state, "offfloor")
        pts = [(m[s], g[s]) for s in ctx["S_offfloor"] if s in m and s in g]
        if pts:
            ax.scatter(*zip(*pts), s=5, alpha=0.3, color="#0072B2", rasterized=True)
        ax.set_title(state_plain(state, fam), fontsize=6.5)
    for j in range(len(panel_keys), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel("teacher-forced fixed-pool margin (nats/token)", fontsize=8)
    fig.supylabel("graded judge score (0-100, 3-draw mean)", fontsize=8)
    fig.suptitle("Off-floor contexts: graded score vs TF margin, per model state", fontsize=9)
    made.append(FG._save(fig, "tfm_margin_vs_graded_scatter", cfg.fig_dir))

    # (3) parent-replication paired dot (machinery-drift check).
    if drift.get("status") in ("ok", "flagged"):
        fig, ax = plt.subplots(figsize=(5.2, 2.2), layout="constrained")
        ax.errorbar(
            drift["rho_parent"],
            1,
            xerr=[[DRIFT_Z * drift["se_parent"]], [DRIFT_Z * drift["se_parent"]]],
            fmt="o",
            color="#888888",
        )
        ax.errorbar(
            drift["rho_new"],
            0,
            xerr=[[DRIFT_Z * drift["se_new"]], [DRIFT_Z * drift["se_new"]]],
            fmt="o",
            color="#0072B2",
        )
        ax.set_yticks(
            [1, 0],
            [
                f"parent run (n={drift['n_parent']})",
                f"this round (n={drift['n_new']})",
            ],
            fontsize=8,
        )
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("rho(TF margin, graded) — sycophancy replication cell, parent subset")
        ax.set_title(
            f"Machinery-drift check: {drift['status']} "
            f"(|delta|={abs(drift['rho_new'] - drift['rho_parent']):.3f} "
            f"vs threshold {drift['threshold']:.3f})",
            fontsize=9,
        )
        made.append(FG._save(fig, "tfm_replication_check", cfg.fig_dir))
    return made


def phase_stats(cfg: Cfg) -> None:
    """V2 (VM): assemble reads, drift gate, lattice, structure reads, figures."""
    G._phase("tfm_stats")
    arms = load_arms(cfg)
    ensure_tfm_config(cfg)
    passes = build_passes(cfg, arms)
    margins = load_margins(cfg, passes)
    reads = stats_reads(cfg, arms, margins)
    expected_reads = 6 if cfg.smoke else 30
    if not cfg.smoke:
        assert len(reads) == expected_reads, (len(reads), expected_reads)
    drift = drift_gate(cfg, reads)
    lattice = lattice_verdict(reads, smoke=cfg.smoke)
    structure = structure_reads(cfg, arms, margins)
    pilot_path = cfg.tfm_dir / "pilot.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
    cfg.stats_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": G._meta(),
        "smoke": cfg.smoke,
        "grain": "per-prompt (NOT #722's cell grain — #722 numbers are recipe precedent only)",
        "n_reads": len(reads),
        "reads": reads,
        "drift_gate": drift,
        "lattice": lattice,
        "structure_reads": structure,
        "pilot": pilot,
        "pool_shas": {
            fam: json.loads(pools_path(cfg, fam).read_text())["content_sha256"]
            for fam in cfg.families()
        },
    }
    out = cfg.stats_dir / "validation_expand.json"
    G._atomic_json(out, payload)
    figs = _figures(cfg, reads, drift, margins, arms)
    if cfg.upload:
        from explore_persona_space.orchestrate import hub

        hub._upload(
            out,
            X.HF_DATA_REPO,
            "dataset",
            f"{cfg.hf_tfm_prefix}/validation_expand.json",
            upload_as_file=True,
            raise_on_error=True,
        )
    print(
        f"[tfm-stats] done: n_reads={len(reads)} drift={drift.get('status')} "
        f"lattice={lattice.get('verdict', lattice.get('status'))} "
        f"figs={[p.name for p in figs]}",
        flush=True,
    )


# ── import-check + main ──────────────────────────────────────────────────────


def _run_import_check() -> None:
    """Execute every deferred import + signature-bind key call shapes (Axis 1)."""
    import inspect

    import matplotlib  # noqa: F401
    import pandas  # noqa: F401
    import torch  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401

    import issue1900_figs as FG
    from explore_persona_space.eval.margin import build_fixed_pairs, compute_tf_margin
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    inspect.signature(compute_tf_margin).bind(object(), object(), object(), [], [], device="cpu")
    inspect.signature(build_fixed_pairs).bind({}, "sycophancy", cap=POOL_SIDE)
    inspect.signature(hub._upload).bind(
        Path("x"), "r", "dataset", "p", upload_as_file=True, raise_on_error=True
    )
    inspect.signature(hub.stage_hub_file).bind("r", "p", Path("x"), repo_type="dataset")
    inspect.signature(hub.stage_hub_prefix).bind("r", "p", Path("x"), repo_type="model")
    inspect.signature(hub.verify_repo_paths_uploaded).bind(
        object(), "r", [], path_in_repo="p", repo_type="dataset"
    )
    inspect.signature(hub.list_hf_files_under_path).bind(
        object(), "r", "p", repo_type="dataset", revision="m"
    )
    inspect.signature(hub.retry_transient).bind(lambda: None, what="probe")
    inspect.signature(assert_out_root_headroom).bind(Path("x"), 1.0, phase="p")
    inspect.signature(R.bootstrap_battery).bind(object(), object(), 10, seed=1)
    inspect.signature(R.observed_rho).bind(object(), object())
    inspect.signature(G._read_raw_rows).bind(object(), "u")
    inspect.signature(FG.arm_plain).bind("cas-pers-con-lr1e5-s42")
    print("[import-check] OK — all deferred imports + call shapes resolve", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=("prep", "gpu", "stats", "all"))
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "data/issue_1900/tfm_work")
    ap.add_argument("--stage-root", type=Path, default=REPO_ROOT / "data/issue_1900/hf_dl")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="imp family x {base, LoRA, FT@2ctx} x 24 contexts; tfm_smoke roots/prefix",
    )
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--worker-slot", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--n-slots", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--pilot-only", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--import-check", action="store_true", help="resolve deferred imports; exit")
    args = ap.parse_args()
    if args.import_check:
        _run_import_check()
        sys.exit(0)
    cfg = Cfg(
        out_root=args.out_root.resolve(),
        stage_root=args.stage_root.resolve(),
        smoke=args.smoke,
        upload=not args.no_upload,
        worker_slot=args.worker_slot,
        n_slots=args.n_slots,
    )
    cfg.tfm_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("prep", "all"):
        phase_prep(cfg)
    if args.phase in ("gpu", "all"):
        phase_gpu(cfg, pilot_only=args.pilot_only)
    if args.phase in ("stats", "all") and cfg.worker_slot is None and not args.pilot_only:
        phase_stats(cfg)
    print("[phase=tfm_done]", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: C-extension finalize rc race (gotchas PyGILState)


if __name__ == "__main__":
    main()
