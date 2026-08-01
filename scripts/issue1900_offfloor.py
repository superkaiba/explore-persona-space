"""#1900 off-floor surface race — same-issue follow-up driver (plan v11).

ONE-variable amendment vs the parent run: the shared 4,000-row judged subset is
replaced by PER-FAMILY subsets stratified on NONZERO BASE graded score
(cas within the already-judged rows; syc slice-extension; imp full-remainder).
Everything else (arms, predictors, anchors, judge instrument, race statistics,
layer 19) is verbatim plan v7 / the parent run record, reused by import.

Phases (each resumable; ``--phase smoke`` chains a scratch end-to-end run):

  f0    (VM)   stage raw_rows + corpus_sample @ CORPUS_PIN; assumption-1
               parquet probe @ PARQUET_PIN; build full-intersection
               judge_inputs shards (base_content + 8 imp/syc arms) via the
               parent P1f writer; emit candidates_{imp,syc}.json.
  f1a   (VM)   base SELECTION pre-pass (Batch API): base_imp x candidates
               (~12.3k) + base_syc x 6,500 slice (seed 19001), 3 draws;
               merge with the parent base rows into
               selection_base_{cas,imp,syc}.json (selection record ONLY —
               these draws never feed P7 or any DV).
  f0b   (VM)   freeze per-family subsets (score_mean > 0 on the selection
               record; syc top-up seed 1900) + the registered composition
               report; membership is FROZEN here (plan section 4 (iv)).
  f1b   (GPU)  leak-through-M guard: 13 refits (m0_L19 + 12 write-maps) with
               train split excluding the SUBSET UNION (parent 4,000 + all
               three off-floor subsets); recompute the 5 map-mediated
               candidate columns; upload maps/ + columns/.
  f2    (VM)   trained-arm judging on NEW rows (merged with parent scores) +
               3 FRESH base ESTIMATION draws on the frozen subsets under a
               fresh work root (the ONLY base scores the race consumes —
               plan section 4 disjoint selection/estimation); upload judge
               text/JSON to the offfloor HF prefix.
  f3    (VM)   rewrite the parent L19 parquets (in_judge_subset per family;
               the 5 map-mediated columns pointed at the F1b refit values,
               parent values kept as *_parentmap record-only columns), then
               run the PRODUCTION race + followup_free entrypoints via
               subprocess against the offfloor dirs.
  figs  (VM)   off-floor figures (offfloor_* stems — never parent stems).

Content hygiene: LMSYS/WildChat prompts + completions are handled digest-only
(counts, shas, scores) — this driver never prints row text.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before numpy/torch: shared-VM thread caps + HF/ANTHROPIC credentials

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1900_gpu as G  # noqa: E402  (Cfg, P1f writer, panel anchors, ccos)
import issue1900_judge as J  # noqa: E402  (judge_unit, build_items, pins)
import issue1900_prep as P0  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1900.offfloor")

ISSUE = 1900
HF_PREFIX = P0.HF_PREFIX  # issue1900_leakrace
OFF_HF_PREFIX = f"{HF_PREFIX}/offfloor"
CORPUS_PIN = P0.CORPUS_PIN
# Parent predictor tables pin (plan v11 section 10; produced 2026-07-31 by P1;
# short revision resolves server-side — probed via HfApi.file_exists 2026-07-31).
PARQUET_PIN = "3bb20deb"
LAYER = 19  # pre-registered primary content layer (plan v7 section 11, verbatim)
D_HIDDEN = 3_584  # Qwen-2.5-7B hidden size (n_train > d refit validity, plan v11)
N_DRAWS = J.N_DRAWS  # 3 (plan v7 section 11 verbatim)
JUDGE_MAX_TOKENS = J.JUDGE_MAX_TOKENS  # 400 (rule 23)
SYC_SLICE_N = 6_500  # plan v11 section 4 (syc pre-pass slice)
SYC_SLICE_SEED = 19_001  # plan v11 section 4 (numpy.random.default_rng(19001))
SYC_TOPUP_SEED = 1_900  # plan v11 section 4 (default_rng(1900) top-up of new nonzero)
SYC_TARGET_N = 4_000  # plan v11 section 4 (n_syc = 4,000)
IMP_LOW_POWER_FLOOR = 500  # plan section 4 F0b: n_imp < 500 => low-power label, proceed
N_VAL = 800  # refit val-split size (parent _judge_splits convention, seed 1900)
FIT_SEED = G.SEED  # 1900 — the parent P1d split seed, verbatim
CONTENT_FAMILIES = ("cas", "imp", "syc")
# The 5 map-mediated candidates -> 7 physical parquet columns (plan section 4 F1b/F3).
MAP_COLS = ("p3a_tc", "p3b_tc", "p3a_ps", "p3b_ps", "p6", "p8a", "p8b")
SMOKE_ARMS = ("imp-pers-con-lr3e5-s42", "syc-pers-po-lr1e5-s42")  # 1 imp + 1 syc
SMOKE_FIT_DIM = 32  # smoke reduces the FEATURE dim, never the code path
SMOKE_EXTRA_ROWS = 900  # non-union rows in the smoke fixture (tr > SMOKE_FIT_DIM)


# ── roots ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Roots:
    """Every output-affecting path for one run mode (production vs smoke)."""

    data: Path  # staging + work dirs (re-downloadable / scratch)
    evalr: Path  # result JSONs (canonical eval_results only in production)
    smoke: bool

    @property
    def config_dir(self) -> Path:  # parent P0 config mirror (subset/arms)
        return REPO_ROOT / "data/issue_1900/config"

    @property
    def parent_judge_dir(self) -> Path:  # committed parent judge scores
        return REPO_ROOT / "eval_results/issue_1900/judge"

    @property
    def parent_race_dir(self) -> Path:
        return REPO_ROOT / "eval_results/issue_1900/race"

    @property
    def stage_root(self) -> Path:  # #1768 mirror root (raw_rows, stores)
        return self.data / "hf_dl"

    @property
    def i1768_root(self) -> Path:
        return self.stage_root / X.HF_PREFIX

    @property
    def inputs_dir(self) -> Path:  # judge_inputs shards (full intersection)
        return self.data / "judge_inputs"

    @property
    def sel_work_dir(self) -> Path:  # F1a selection judge work root
        return self.data / "judge_work" / "selection"

    @property
    def est_work_dir(self) -> Path:  # F2 estimation judge work root (FRESH)
        return self.data / "judge_work" / "estimation"

    @property
    def cfg_out(self) -> Path:  # candidates/subsets/composition (committed)
        return self.evalr / "config"

    @property
    def judge_out(self) -> Path:
        return self.evalr / "judge"

    @property
    def race_out(self) -> Path:
        return self.evalr / "race"

    @property
    def p1_root(self) -> Path:  # rewritten predictor tables + validation
        return self.data / "p1_root"

    @property
    def gpu_out(self) -> Path:  # F1b out root (maps/, columns/, anchors/)
        return self.data / "out"

    @property
    def columns_dir(self) -> Path:
        return self.gpu_out / "columns"

    @property
    def off_prefix(self) -> str:
        """HF prefix for this run's uploads (smoke -> scratch smoke_probe)."""
        return f"{OFF_HF_PREFIX}/smoke_probe" if self.smoke else OFF_HF_PREFIX

    @property
    def fig_dir(self) -> Path:
        return (self.data / "figs") if self.smoke else (REPO_ROOT / "figures/issue_1900")


def make_roots(smoke: bool) -> Roots:
    if smoke:
        base = REPO_ROOT / "data/issue_1900/offfloor_smoke"
        return Roots(data=base, evalr=base / "eval", smoke=True)
    return Roots(
        data=REPO_ROOT / "data/issue_1900/offfloor",
        evalr=REPO_ROOT / "eval_results/issue_1900/offfloor",
        smoke=False,
    )


def _meta() -> dict:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": J._git_commit(),
        "numpy": np.__version__,
        "issue": ISSUE,
        "corpus_pin": CORPUS_PIN,
        "parquet_pin": PARQUET_PIN,
        "script": "scripts/issue1900_offfloor.py",
    }


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _data_repo() -> str:
    return X.HF_DATA_REPO


# ── shared loaders ───────────────────────────────────────────────────────────


def content_arm_entries(roots: Roots) -> list[dict]:
    arms = J.load_arms(roots.config_dir)
    content = [a for a in arms if a["kind"] == "content"]
    assert len(content) == 12, len(content)
    return content


def load_intersection(roots: Roots) -> list[str]:
    payload = json.loads((roots.cfg_out / "candidates_imp.json").read_text())
    return list(payload["intersection_shas"])


def load_candidates(roots: Roots, fam: str) -> list[str]:
    payload = json.loads((roots.cfg_out / f"candidates_{fam}.json").read_text())
    return list(payload["shas"])


def load_offfloor_subset(roots: Roots, fam: str) -> list[str]:
    payload = json.loads((roots.cfg_out / f"subset_{fam}.json").read_text())
    assert payload["n"] == len(payload["shas"]), (fam, payload["n"])
    return list(payload["shas"])


def _scored_rows(payload: dict) -> list[dict]:
    return [r for r in payload["rows"] if r["score_mean"] is not None]


def nonzero_shas(payload: dict) -> list[str]:
    """Selection rule: score_mean > 0 over SCORED rows (plan section 4)."""
    return [r["sha"] for r in _scored_rows(payload) if r["score_mean"] > 0]


def assert_pins_equal(a: dict, b: dict, what: str) -> None:
    """Rule-18 instrument pins must match before ANY score merge (plan section 8)."""
    for key in ("judge_model", "n_draws", "max_tokens", "rubric_sha256"):
        assert a["judge"][key] == b["judge"][key], (what, key, a["judge"][key], b["judge"][key])


# ── subset construction (pure, seeded — pinned by tests) ────────────────────


def draw_syc_slice(candidates: list[str], n: int = SYC_SLICE_N, seed: int = SYC_SLICE_SEED):
    """Seeded without-replacement slice of the remaining intersection (plan section 4)."""
    cand = sorted(candidates)
    assert len(cand) == len(set(cand)), "duplicate candidate shas"
    if len(cand) <= n:
        return cand
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.asarray(cand, dtype=object), size=n, replace=False)
    return sorted(str(s) for s in picked)


def build_subsets(
    sel_records: dict[str, dict],
    parent_subset: set[str],
    *,
    syc_target: int = SYC_TARGET_N,
    topup_seed: int = SYC_TOPUP_SEED,
) -> tuple[dict[str, list[str]], dict]:
    """Freeze the per-family subsets from the SELECTION record (plan section 4).

    cas: nonzero within the parent 4,000 (zero new judging). imp: ALL nonzero
    (parent + pre-pass). syc: all parent nonzero + a seeded top-up of NEW
    nonzero to n_syc = 4,000 (shortfall -> take all, reported). Membership is
    frozen here; estimation draws never re-select (plan section 4 (iv)).
    Returns (subsets, composition_report).
    """
    subsets: dict[str, list[str]] = {}
    report: dict = {"rule": "score_mean > 0 on the SELECTION record", "per_family": {}}
    for fam in CONTENT_FAMILIES:
        payload = sel_records[fam]
        nz = nonzero_shas(payload)
        assert len(nz) == len(set(nz)), (fam, "duplicate shas in selection record")
        if fam == "cas":
            keep = sorted(s for s in nz if s in parent_subset)
        elif fam == "imp":
            keep = sorted(nz)
        else:  # syc: parent nonzero + seeded top-up of the NEW nonzero
            parent_nz = sorted(s for s in nz if s in parent_subset)
            new_nz = sorted(s for s in nz if s not in parent_subset)
            need = syc_target - len(parent_nz)
            assert need >= 0, (len(parent_nz), syc_target)
            if len(new_nz) <= need:
                topup = new_nz  # shortfall: take all, report realized n
            else:
                rng = np.random.default_rng(topup_seed)
                topup = sorted(
                    str(s)
                    for s in rng.choice(np.asarray(new_nz, dtype=object), size=need, replace=False)
                )
            keep = sorted(parent_nz + topup)
        subsets[fam] = keep
        keep_set = set(keep)
        sel_rows = [r for r in _scored_rows(payload) if r["sha"] in keep_set]
        single = sum(1 for r in sel_rows if sum(1 for s in r["kept_draw_scores"] if s > 0) == 1)
        le5 = sum(1 for r in sel_rows if r["score_mean"] <= 5)
        report["per_family"][fam] = {
            "n": len(keep),
            "n_scored_selection_rows": len(sel_rows),
            "single_draw_nonzero": single,
            "mean_le5": le5,
            "n_parent_rows": sum(1 for s in keep if s in parent_subset),
            "n_new_rows": sum(1 for s in keep if s not in parent_subset),
        }
    report["low_power_imp"] = len(subsets["imp"]) < IMP_LOW_POWER_FLOOR
    return subsets, report


# ── F0: staging + candidates + judge inputs ──────────────────────────────────

F0_UNITS_BEH = {"imp": 4, "syc": 4}  # content arm families needing NEW judging


def _f0_units(roots: Roots) -> list[str]:
    arms = content_arm_entries(roots)
    units = [a["arm_id"] for a in arms if a["beh_key"] in F0_UNITS_BEH]
    assert len(units) == 8, units
    if roots.smoke:
        units = [u for u in units if u in SMOKE_ARMS]
        assert len(units) == 2, units
    return ["base_content", *units]


def _stage_raw_rows(roots: Roots, unit: str, max_shards: int | None) -> None:
    """Per-file scoped staging of a unit's raw_rows shards @ CORPUS_PIN (#833)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dest = roots.i1768_root / "corpus_capture" / unit
    if list(dest.glob("raw_rows_*.jsonl")):
        return
    listing = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            HfApi(),
            _data_repo(),
            f"{X.HF_PREFIX}/corpus_capture/{unit}",
            repo_type="dataset",
            revision=CORPUS_PIN,
        ),
        what=f"raw_rows scoped listing ({unit})",
    )
    shards = sorted(p for p in listing if "raw_rows_" in Path(p).name and p.endswith(".jsonl"))
    assert shards, (unit, "no raw_rows shards at the pin")
    if max_shards is not None:
        shards = shards[:max_shards]
    for p in shards:
        hub.stage_hub_file(
            _data_repo(), p, dest / Path(p).name, repo_type="dataset", revision=CORPUS_PIN
        )
    logger.info("[f0] %s: staged %d raw_rows shards", unit, len(shards))


def _stage_corpus_sample(roots: Roots) -> dict[str, str]:
    from explore_persona_space.orchestrate import hub

    local = roots.i1768_root / "inputs" / "corpus_sample.json"
    if not local.exists():
        hub.stage_hub_file(
            _data_repo(),
            f"{X.HF_PREFIX}/inputs/corpus_sample.json",
            local,
            repo_type="dataset",
            revision=CORPUS_PIN,
        )
    sample = json.loads(local.read_text())
    return {r["sha"]: r["prompt"] for r in sample["rows"]}


def _stage_parent_parquet(roots: Roots, arm_id: str) -> Path:
    """Parent predictor table @ PARQUET_PIN into a NON-rebinding parent dir."""
    from explore_persona_space.orchestrate import hub

    dest = roots.p1_root / "parent_tables" / f"{arm_id}_L{LAYER}.parquet"
    if not dest.exists():
        hub.stage_hub_file(
            _data_repo(),
            f"{HF_PREFIX}/predictor_tables/{arm_id}_L{LAYER}.parquet",
            dest,
            repo_type="dataset",
            revision=PARQUET_PIN,
        )
    return dest


def _assumption1_probe(roots: Roots) -> list[str]:
    """Plan section 12.1: the pinned parquet covers the FULL intersection with
    non-null candidate columns on non-subset rows. Returns the intersection."""
    import pandas as pd

    tab = pd.read_parquet(_stage_parent_parquet(roots, "imp-pers-con-lr3e5-s42"))
    n = len(tab)
    assert n >= 0.9 * 16_318, (n, "parquet does not cover the full intersection")
    parent = set(J.load_subset(roots.config_dir))
    non_subset = tab[~tab["sha"].isin(parent)]
    sample = non_subset.sample(n=min(100, len(non_subset)), random_state=FIT_SEED)
    check_cols = ["p1_tc", "p2_tc", "p3a_tc", "p3b_tc", "p4_tc", "p5", "p6", "p8a", "p8b", "p9_k16"]
    for col in check_cols:
        n_null = int(sample[col].isna().sum())
        assert n_null == 0, (col, n_null, "null candidate values on non-subset shas")
    logger.info("[f0] assumption-1 probe PASS: n_rows=%d, 100-sha non-null check", n)
    # The #1768 corpus carries duplicate-sha rows (16,400 rows / 16,318 unique;
    # gotchas real-corpus-dupes class). The parent P0 subset draw DEDUPED, so
    # the parent frames never saw a dup row (verified: 0 dup rows in the
    # parent 4,000). Mirror that posture: EXCLUDE duplicated shas from the
    # candidate pool so off-floor frames stay sha-unique (mechanism untouched).
    counts = tab["sha"].value_counts()
    dup_shas = set(counts[counts > 1].index)
    assert not (dup_shas & parent), "parent subset unexpectedly carries duplicated shas"
    shas = [s for s in dict.fromkeys(tab["sha"]) if s not in dup_shas]
    logger.info(
        "[f0] intersection: %d unique shas (%d duplicated shas excluded, %d dup rows)",
        len(shas),
        len(dup_shas),
        int(counts[counts > 1].sum()),
    )
    assert len(shas) >= 0.9 * 16_318, len(shas)
    return shas


def phase_f0(roots: Roots) -> None:
    _phase("f0_prep")
    parent_subset = J.load_subset(roots.config_dir)
    intersection = _assumption1_probe(roots)
    prompt_by_sha = _stage_corpus_sample(roots)
    units = _f0_units(roots)
    max_shards = 1 if roots.smoke else None
    for unit in units:
        _stage_raw_rows(roots, unit, max_shards)

    # judge_inputs shards over the FULL intersection (smoke: staged-row slice),
    # via the parent P1f writer (schema-exact; done-sentinel resume).
    cfg = G.Cfg(out_root=roots.data, stage_root=roots.stage_root, smoke=False, upload=False)
    rows_scope = intersection
    if roots.smoke:
        base_raw = G._read_raw_rows(cfg, "base_content")
        rows_scope = [s for s in intersection if s in base_raw][:600]
        assert rows_scope, "smoke: no staged base rows intersect the parquet"
    for k, unit in enumerate(units):
        shards = G.run_p1f_unit(cfg, unit, rows_scope, prompt_by_sha)
        cov = json.loads((roots.inputs_dir / f"{unit}.done.json").read_text())["n_rows"]
        if not roots.smoke:
            assert cov >= 0.99 * len(intersection), (unit, cov, "sha-join coverage < 0.99")
        print(f"[f0] unit {k + 1}/{len(units)} {unit} shards={len(shards)} rows={cov}", flush=True)

    candidates = sorted(set(rows_scope) - set(parent_subset))
    syc_slice = draw_syc_slice(candidates)
    if roots.smoke:  # tiny slice for the dry-run items build
        syc_slice = syc_slice[: min(200, len(syc_slice))]
        candidates = candidates[: min(200, len(candidates))]
    _atomic_json(
        roots.cfg_out / "candidates_imp.json",
        {
            "meta": _meta(),
            "n": len(candidates),
            "shas": candidates,
            "rule": "full remaining intersection (intersection - parent subset)",
            # ALWAYS the full parquet row set (smoke included): the F1b fixture
            # + F3 rewrite key on it; the subsets stay parent-4,000-contained.
            "intersection_shas": intersection,
            "n_intersection": len(intersection),
        },
    )
    _atomic_json(
        roots.cfg_out / "candidates_syc.json",
        {
            "meta": _meta(),
            "n": len(syc_slice),
            "shas": syc_slice,
            "rule": f"default_rng({SYC_SLICE_SEED}) slice of {SYC_SLICE_N} from the remainder",
        },
    )
    print(
        f"[f0] done: intersection={len(rows_scope)} cand_imp={len(candidates)} "
        f"cand_syc={len(syc_slice)}",
        flush=True,
    )


# ── judge plumbing (F1a + F2 share it) ───────────────────────────────────────


def _judge_with_resume(
    roots: Roots,
    *,
    tag: str,
    beh_key: str,
    unit: str,
    shas: list[str],
    work_dir: Path,
    out_path: Path,
    dry_run: bool,
) -> dict:
    """One judge unit with the run_units resume-skip contract (parent P2)."""
    rows = J.load_judge_inputs(roots.inputs_dir, unit)
    items, id_map = J.build_items(shas, rows, None)
    if not roots.smoke:
        missing = len(shas) - len(items)
        assert missing <= 0.01 * len(shas), (tag, missing, "judge-input coverage < 0.99")
    if out_path.exists() and not dry_run:
        prior = json.loads(out_path.read_text())
        if (
            prior.get("n_items") == len(items)
            and prior.get("judge", {}).get("n_draws") == N_DRAWS
            and prior.get("judge", {}).get("max_tokens") == JUDGE_MAX_TOKENS
        ):
            print(f"[judge] {tag} resume-skip (n_items={len(items)})", flush=True)
            return prior
    return J.judge_unit(
        tag,
        beh_key,
        items,
        id_map,
        work_dir,
        out_path,
        n_draws=N_DRAWS,
        max_tokens=JUDGE_MAX_TOKENS,
        dry_run=dry_run,
    )


def _merged_payload(
    parent: dict, new: dict | None, keep_shas: set[str], unit_tag: str, note: str
) -> dict:
    """Parent rows filtered to keep_shas + NEW rows, aggregates recomputed."""
    if new is not None:
        assert_pins_equal(parent, new, unit_tag)
    rows = [r for r in parent["rows"] if r["sha"] in keep_shas]
    seen = {r["sha"] for r in rows}
    if new is not None:
        extra = [r for r in new["rows"] if r["sha"] in keep_shas and r["sha"] not in seen]
        rows = rows + extra
    assert len({r["sha"] for r in rows}) == len(rows), (unit_tag, "duplicate merged shas")
    means = np.asarray(
        [r["score_mean"] for r in rows if r["score_mean"] is not None], dtype=np.float64
    )
    payload = {
        "meta": {**_meta(), "unit": unit_tag, "merge_note": note},
        "judge": parent["judge"],
        "beh_key": parent["beh_key"],
        "n_items": len(rows),
        "n_scored_items": int(len(means)),
        "n_all_draws_dropped_items": int(len(rows) - len(means)),
        "sd_context_means": float(np.std(means, ddof=1)) if len(means) > 1 else None,
        "share_ge10": float(np.mean(means >= 10.0)) if len(means) else None,
        "sources": {
            "parent_rows": sum(1 for r in rows if r["sha"] in {x["sha"] for x in parent["rows"]}),
            "new_rows": len(rows)
            - sum(1 for r in rows if r["sha"] in {x["sha"] for x in parent["rows"]}),
        },
        "rows": rows,
    }
    return payload


def _parent_scores(roots: Roots, name: str) -> dict:
    path = roots.parent_judge_dir / f"arm_scores_{name}.json"
    assert path.exists(), path
    return json.loads(path.read_text())


def phase_f1a(roots: Roots, dry_run: bool) -> None:
    _phase("f1a_selection_judge")
    specs = [
        ("sel_base_imp", "imp", load_candidates(roots, "imp")),
        ("sel_base_syc", "syc", load_candidates(roots, "syc")),
    ]
    tmp_payloads: dict[str, dict] = {}
    for tag, beh, shas in specs:
        tmp_payloads[beh] = _judge_with_resume(
            roots,
            tag=tag,
            beh_key=beh,
            unit="base_content",
            shas=shas,
            work_dir=roots.sel_work_dir,
            out_path=roots.judge_out / f"arm_scores_{tag}.json",
            dry_run=dry_run,
        )
    if dry_run:
        print("[f1a] dry-run: items built + validated, 0 API calls", flush=True)
        return
    # Merge with the parent base rows into the SELECTION record (record only).
    for fam in CONTENT_FAMILIES:
        parent = _parent_scores(roots, f"base_{fam}")
        new = None
        keep = {r["sha"] for r in parent["rows"]}
        if fam in tmp_payloads and not tmp_payloads[fam].get("dry_run"):
            new = json.loads((roots.judge_out / f"arm_scores_sel_base_{fam}.json").read_text())
            keep |= {r["sha"] for r in new["rows"]}
        merged = _merged_payload(
            parent,
            new,
            keep,
            f"selection_base_{fam}",
            "SELECTION record only — never feeds P7 or any DV",
        )
        _atomic_json(roots.judge_out / f"selection_base_{fam}.json", merged)
    _atomic_json(
        roots.judge_out / "selection_done.json",
        {
            **_meta(),
            "families": list(CONTENT_FAMILIES),
            "n_new_units": len(specs),
            "dry_run": False,
        },
    )
    _upload_judge_text(roots)
    print("[f1a] done: selection_base_{cas,imp,syc}.json written", flush=True)


def phase_f0b(roots: Roots) -> None:
    _phase("f0b_select")
    parent_subset = set(J.load_subset(roots.config_dir))
    sel_records = {
        fam: json.loads((roots.judge_out / f"selection_base_{fam}.json").read_text())
        for fam in CONTENT_FAMILIES
    }
    subsets, report = build_subsets(sel_records, parent_subset)
    # smoke: the cached-judge subsets are parent-4,000-contained, which the
    # full parquet always covers — no scope restriction needed (MIN_ROWS=50).
    for fam, shas in subsets.items():
        _atomic_json(
            roots.cfg_out / f"subset_{fam}.json",
            {
                "meta": _meta(),
                "family": fam,
                "n": len(shas),
                "shas": shas,
                "rule": "nonzero base graded score (selection record); membership FROZEN",
                "seeds": {"syc_slice": SYC_SLICE_SEED, "syc_topup": SYC_TOPUP_SEED},
            },
        )
    _atomic_json(roots.cfg_out / "composition_report.json", {"meta": _meta(), **report})
    print(
        f"[f0b] done: n_cas={len(subsets['cas'])} n_imp={len(subsets['imp'])} "
        f"n_syc={len(subsets['syc'])} low_power_imp={report['low_power_imp']}",
        flush=True,
    )


# ── F2: trained-arm + base estimation judging ────────────────────────────────


def phase_f2(roots: Roots, dry_run: bool) -> None:
    _phase("f2_judge")
    # rule 24(ii) isolation: the estimation work root is DISJOINT from the
    # selection root, so no #1019 checkpoint can serve back a selection draw.
    est, sel = roots.est_work_dir.resolve(), roots.sel_work_dir.resolve()
    assert est != sel and not est.is_relative_to(sel) and not sel.is_relative_to(est), (est, sel)
    assert not list(est.glob("sel_*")), "selection checkpoints found under the estimation root"
    subsets = {f: load_offfloor_subset(roots, f) for f in CONTENT_FAMILIES}
    arms = content_arm_entries(roots)
    if roots.smoke:
        arms = [a for a in arms if a["arm_id"] in SMOKE_ARMS]

    # (i) trained arms: judge NEW rows only; merge with parent rows on overlap.
    for k, arm in enumerate(arms):
        fam = arm["beh_key"]
        subset = set(subsets[fam])
        parent = _parent_scores(roots, arm["arm_id"])
        parent_shas = {r["sha"] for r in parent["rows"]}
        new_shas = sorted(subset - parent_shas)
        new = None
        if fam in F0_UNITS_BEH and new_shas and not roots.smoke:
            new = _judge_with_resume(
                roots,
                tag=f"f2_{arm['arm_id']}",
                beh_key=fam,
                unit=arm["arm_id"],
                shas=new_shas,
                work_dir=roots.sel_work_dir.parent / "trained",
                out_path=roots.judge_out / f"arm_scores_f2_{arm['arm_id']}.json",
                dry_run=dry_run,
            )
        elif roots.smoke and fam in F0_UNITS_BEH:
            # smoke: exercise the ARM-unit judge_inputs load + items build +
            # dry-run judge path (subset rows; build_items drops unstaged shas)
            _judge_with_resume(
                roots,
                tag=f"f2_{arm['arm_id']}",
                beh_key=fam,
                unit=arm["arm_id"],
                shas=new_shas or sorted(subset),
                work_dir=roots.sel_work_dir.parent / "trained",
                out_path=roots.judge_out / f"arm_scores_f2_{arm['arm_id']}.json",
                dry_run=True,
            )
        if dry_run:
            continue
        merged = _merged_payload(
            parent,
            None if (new is None or new.get("dry_run")) else new,
            subset,
            arm["arm_id"],
            "off-floor: parent rows filtered to subset + new-row scores (same instrument)",
        )
        if not roots.smoke and fam in F0_UNITS_BEH:
            n_scored = merged["n_scored_items"]
            assert n_scored >= 0.95 * len(subset), (arm["arm_id"], n_scored, len(subset))
        _atomic_json(roots.judge_out / f"arm_scores_{arm['arm_id']}.json", merged)
        print(
            f"[f2] arm {k + 1}/{len(arms)} {arm['arm_id']} n={merged['n_items']} "
            f"new={len(new_shas)}",
            flush=True,
        )
    if dry_run:
        print("[f2] dry-run: trained items built, 0 API calls", flush=True)
        return

    # (ii) base ESTIMATION: 3 FRESH draws on the frozen subsets, fresh work root
    # (rule 24(ii)); the ONLY base scores the race consumes.
    for fam in CONTENT_FAMILIES:
        tag = f"est_base_{fam}"
        out_path = roots.judge_out / f"arm_scores_base_{fam}.json"
        assert not str(roots.est_work_dir).startswith(str(roots.sel_work_dir)), "work-dir overlap"
        if roots.smoke:
            # cached-judge smoke: the estimation file is a labeled parent copy;
            # the live estimation path is exercised via the dry-run items build.
            _judge_with_resume(
                roots,
                tag=tag,
                beh_key=fam,
                unit="base_content",
                shas=subsets[fam],
                work_dir=roots.est_work_dir,
                out_path=out_path,
                dry_run=True,
            )
            parent = _parent_scores(roots, f"base_{fam}")
            merged = _merged_payload(
                parent,
                None,
                set(subsets[fam]),
                f"base_{fam}",
                "SMOKE cached-judge: parent base rows as estimation stand-in",
            )
            _atomic_json(out_path, merged)
            continue
        payload = _judge_with_resume(
            roots,
            tag=tag,
            beh_key=fam,
            unit="base_content",
            shas=subsets[fam],
            work_dir=roots.est_work_dir,
            out_path=out_path,
            dry_run=False,
        )
        n_scored = payload["n_scored_items"]
        assert n_scored >= 0.95 * len(subsets[fam]), (fam, n_scored, len(subsets[fam]))
        print(f"[f2] estimation base_{fam} n={payload['n_items']} scored={n_scored}", flush=True)

    _atomic_json(
        roots.judge_out / "judge_done.json",
        {
            **_meta(),
            "families": list(CONTENT_FAMILIES),
            "arms": [a["arm_id"] for a in arms],
            "estimation": "fresh 3-draw base pass on frozen subsets (disjoint from selection)",
        },
    )
    _upload_judge_text(roots)
    print("[f2] done", flush=True)


def _upload_judge_text(roots: Roots) -> None:
    """Judge JSONs + raw reasoning texts -> the offfloor HF prefix (text always)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    local = roots.judge_out
    files = sorted(p for p in local.rglob("*.json") if not p.name.endswith(".tmp"))
    if not files:
        return
    prefix = f"{roots.off_prefix}/judge"
    hub._upload(
        local, repo_id=_data_repo(), repo_type="dataset", path_in_repo=prefix, raise_on_error=True
    )
    expected = [f"{prefix}/{p.relative_to(local)}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        api, _data_repo(), expected, path_in_repo=prefix, repo_type="dataset"
    )
    assert not missing, f"judge upload verify FAILED: missing {missing}"
    for work in (roots.sel_work_dir, roots.sel_work_dir.parent / "trained", roots.est_work_dir):
        if work.exists() and any(work.rglob("judge_raw.json")):
            J._upload_raw(work, path_in_repo=f"{roots.off_prefix}/judge/raw/{work.name}")
    logger.info("[upload] judge JSONs verified at %s (%d files)", prefix, len(files))


# ── F1b: leak-through-M refits + map-mediated column recompute (GPU) ─────────


def _resolve_device(arg: str):
    import torch

    if arg == "cuda":
        assert torch.cuda.is_available(), "f1b production is GPU-only (plan section 9)"
    return torch.device(arg)


def _offfloor_splits(shas: list[str], exclude: set[str], te_shas: set[str], d: int):
    """(tr, val, te): tr/val exclude the SUBSET UNION (leak-through-M guard);
    te = the off-floor rows this refit's reads serve. Seed-1900 val split
    (parent _judge_splits convention, verbatim)."""
    te = np.asarray([i for i, s in enumerate(shas) if s in te_shas])
    rest = np.asarray([i for i, s in enumerate(shas) if s not in exclude])
    rng = np.random.default_rng(FIT_SEED)
    perm = rng.permutation(len(rest))
    n_val = min(N_VAL, max(1, len(rest) // 4))  # adaptive: production == 800
    val = rest[perm[:n_val]]
    tr = rest[perm[n_val:]]
    assert len(te) > 0, "empty te (off-floor rows missing from the store)"
    assert len(tr) > d, (len(tr), d, "n_train > d refit validity (plan section 11)")
    return tr, val, te


def _fit_persist(roots: Roots, name: str, Xd, Yd, tr, val, te, dev) -> Path:
    """One `_fit_map` refit + identity+bias/kNN reads (parent P1d, device-param)."""
    import torch

    import issue1768_fit as F

    out_pt = roots.gpu_out / "maps" / f"{name}.pt"
    out_js = roots.gpu_out / "maps" / f"{name}.json"
    if out_pt.exists() and out_js.exists():
        return out_pt
    t0 = time.time()
    pred_te, meta, payload = F._fit_map(Xd, Yd, tr, val, te, dev)
    reads = F._map_reads(pred_te, Yd[te])
    ib = F._identity_bias_reads(Xd[tr], Yd[tr], Xd[te], Yd[te])
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pt.with_suffix(".pt.tmp")
    torch.save({"payload": payload, "name": name}, tmp)
    os.replace(tmp, out_pt)
    _atomic_json(
        out_js,
        {
            "name": name,
            "n_tr": int(len(tr)),
            "n_val": int(len(val)),
            "n_te": int(len(te)),
            "fit_meta": {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)},
            "reads": reads,
            "identity_bias": ib,
            "split": "tr/val exclude the SUBSET UNION (parent 4,000 + off-floor subsets), "
            "seed 1900; te = off-floor rows",
            "elapsed_s": round(time.time() - t0, 1),
            **_meta(),
        },
    )
    print(
        f"[f1b] fit {name} n_tr={len(tr)} n_te={len(te)} elapsed={time.time() - t0:.0f}s",
        flush=True,
    )
    return out_pt


def _stage_f1b_inputs(roots: Roots, arms: list[dict]) -> None:
    """Per-file scoped staging of the fit + recompute inputs @ their pins."""
    from explore_persona_space.orchestrate import hub

    _phase("f1b_stage")
    root = roots.i1768_root
    files = [("corpus_capture/base_content/pooled.pt", root)]
    for a in arms:
        files.append((f"corpus_capture/{a['arm_id']}/pooled.pt", root))
        files.append((f"corpus_capture_tf/{a['arm_id']}/pooled_tf.pt", root))
    for beh in sorted({a["beh_key"] for a in arms}):
        files.append((f"panel_capture/base_{beh}/pooled.pt", root))
    files.append(("inputs/corpus_sample.json", root))
    files.append(("arm_registry.json", root))
    for rel, dest_root in files:
        local = dest_root / rel
        if not local.exists():
            hub.stage_hub_file(
                _data_repo(),
                f"{X.HF_PREFIX}/{rel}",
                local,
                repo_type="dataset",
                revision=CORPUS_PIN,
            )
    # parent anchors (issue1900_leakrace/anchors/{mix}.pt — no revision pin; the
    # anchors are parent P1b outputs, single-generation per plan section 10)
    for mix in sorted({a["mix_arm_id"] for a in arms}):
        local = roots.gpu_out / "anchors" / f"{mix}.pt"
        if not local.exists():
            hub.stage_hub_file(
                _data_repo(), f"{HF_PREFIX}/anchors/{mix}.pt", local, repo_type="dataset"
            )
    logger.info(
        "[f1b] staging complete (%d store files, %d mixes)",
        len(files),
        len({a["mix_arm_id"] for a in arms}),
    )


def recompute_mapcols(
    roots: Roots, entry: dict, m0_payload, wmap_payload, rb: dict, dev, *, panel_anchor: bool
) -> Path:
    """The 5 map-mediated candidate columns (7 physical cols) with REFIT maps.

    Mirrors the parent run_p1e_table map-column block verbatim (centering =
    full-cell mean-mapped row mean); writes sha + MAP_COLS for EVERY cell row.
    """
    import pandas as pd
    import torch

    import issue1768_fit as F
    import issue779_ffc_n1m_fits as n1m

    arm_id = entry["arm_id"]
    out = roots.columns_dir / f"{arm_id}_L{LAYER}_mapcols.parquet"
    if out.exists():
        return out
    cell = F.load_corpus_cell(arm_id, LAYER, roots.i1768_root)
    c0 = cell["C0"]
    anc = torch.load(
        roots.gpu_out / "anchors" / f"{entry['mix_arm_id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    a_ctx = np.asarray(anc["A_ctx"][LAYER].numpy(), dtype=np.float64)
    a_ans = np.asarray(anc["A_ans"][LAYER].numpy(), dtype=np.float64)
    assert a_ctx.shape == (c0.shape[1],), (a_ctx.shape, c0.shape)

    def _apply(payload, rows: np.ndarray) -> np.ndarray:
        return np.asarray(n1m.apply_map(payload, rows, dev), dtype=np.float64)

    cfg = G.Cfg(out_root=roots.gpu_out, stage_root=roots.stage_root, upload=False)
    ps = G._panel_anchor(cfg, entry, LAYER) if panel_anchor else None
    mpred = _apply(m0_payload, c0)
    assert mpred.shape == c0.shape, (mpred.shape, c0.shape)
    mbar = mpred.mean(axis=0)
    m0_actx = _apply(m0_payload, a_ctx[None, :])[0]
    df = pd.DataFrame({"sha": cell["sha"]})
    df["p3a_tc"] = G._ccos(mpred, m0_actx, mbar)
    df["p3b_tc"] = G._ccos(mpred, a_ans, mbar)
    if ps is not None:
        df["p3a_ps"] = G._ccos(mpred, _apply(m0_payload, ps[0][None, :])[0], mbar)
        df["p3b_ps"] = G._ccos(mpred, ps[1], mbar)
    else:
        df["p3a_ps"] = np.nan
        df["p3b_ps"] = np.nan
    df["p6"] = (mpred - mbar) @ rb[entry["beh_key"]][LAYER]
    wpred = _apply(wmap_payload, c0)
    assert wpred.shape == c0.shape, (wpred.shape, c0.shape)
    df["p8a"] = np.linalg.norm(wpred, axis=1)
    df["p8b"] = G._ccos(wpred, rb[entry["beh_key"]][LAYER], None)
    # Producer-side sha-dedup (r1 Critical 1 sibling): keep the mapcols
    # parquet sha-unique so the F3 rewrite join is well-defined regardless of
    # producer version; keep="first" is value-safe (dup shas never in subsets).
    df = df[~df["sha"].duplicated(keep="first")].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def phase_f1b(roots: Roots, device: str) -> None:
    import issue1768_fit as F

    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    _phase("f1b_refits")
    if not roots.smoke:
        free = assert_out_root_headroom(roots.gpu_out, 25.0, phase="f1b")
        logger.info("[f1b] headroom OK: %.1f GB free at %s", free, roots.gpu_out)
    dev = _resolve_device(device)
    arms = content_arm_entries(roots)
    if roots.smoke:
        arms = [a for a in arms if a["arm_id"] in SMOKE_ARMS]
    parent_subset = set(J.load_subset(roots.config_dir))
    subsets = {f: set(load_offfloor_subset(roots, f)) for f in CONTENT_FAMILIES}
    union = parent_subset | subsets["cas"] | subsets["imp"] | subsets["syc"]
    off_union = subsets["cas"] | subsets["imp"] | subsets["syc"]
    logger.info("[f1b] union=%d off_union=%d", len(union), len(off_union))
    if roots.smoke:
        _build_smoke_fixture(roots, arms)
    else:
        _stage_f1b_inputs(roots, arms)
    d_fit = SMOKE_FIT_DIM if roots.smoke else D_HIDDEN

    # fit 1/13: m0 @ L19 (base context -> base answer), union-excluded split
    c0, v0, shas = G._base_matrices(
        G.Cfg(out_root=roots.gpu_out, stage_root=roots.stage_root, upload=False),
        "base_content",
        LAYER,
    )
    assert c0.shape == v0.shape and c0.shape[1] == d_fit, (c0.shape, v0.shape, d_fit)
    tr, val, te = _offfloor_splits(shas, union, off_union, d_fit)
    _fit_persist(roots, f"m0_L{LAYER}", c0, v0, tr, val, te, dev)

    # fits 2..13: write-maps per content arm (siblings' C0 -> delta v_tf),
    # union-excluded sibling pool; te = the target arm's off-floor family rows
    for k, entry in enumerate(arms):
        arm_id = entry["arm_id"]
        siblings = [
            a["arm_id"] for a in arms if a["beh_key"] == entry["beh_key"] and a["arm_id"] != arm_id
        ]
        if not roots.smoke:
            assert len(siblings) == 3, (arm_id, siblings)
        blocks_x, blocks_y = [], []
        for sib in siblings:
            cell = F.load_corpus_cell(sib, LAYER, roots.i1768_root)
            keep = np.asarray([i for i, s in enumerate(cell["sha"]) if s not in union])
            blocks_x.append(cell["C0"][keep])
            blocks_y.append((cell["Vplus_tf"] - cell["V0"])[keep])
        tgt = F.load_corpus_cell(arm_id, LAYER, roots.i1768_root)
        fam_rows = subsets[entry["beh_key"]]
        tgt_te = np.asarray([i for i, s in enumerate(tgt["sha"]) if s in fam_rows])
        if not siblings:  # smoke-only degenerate self-sibling (parent convention)
            logger.warning("[f1b] SMOKE wmap %s: sibling set degenerates to self", arm_id)
            keep = np.asarray([i for i, s in enumerate(tgt["sha"]) if s not in union])
            blocks_x.append(tgt["C0"][keep])
            blocks_y.append((tgt["Vplus_tf"] - tgt["V0"])[keep])
        Xd = np.vstack(blocks_x + [tgt["C0"][tgt_te]])
        Yd = np.vstack(blocks_y + [(tgt["Vplus_tf"] - tgt["V0"])[tgt_te]])
        n_pool = sum(b.shape[0] for b in blocks_x)
        rng = np.random.default_rng(FIT_SEED)
        perm = rng.permutation(n_pool)
        n_val = min(N_VAL, max(1, n_pool // 4))  # adaptive: production == 800
        val = perm[:n_val]
        tr = perm[n_val:]
        te = np.arange(n_pool, n_pool + len(tgt_te))
        assert len(tr) > d_fit, (len(tr), d_fit, "wmap n_train > d")
        _fit_persist(roots, f"wmap_{arm_id}_L{LAYER}", Xd, Yd, tr, val, te, dev)
        print(f"[f1b] wmap {k + 1}/{len(arms)} {arm_id} done", flush=True)

    _phase("f1b_columns")
    rb = _load_rb(roots, arms)
    m0_payload = G._load_map_payload(
        G.Cfg(out_root=roots.gpu_out, stage_root=roots.stage_root, upload=False), f"m0_L{LAYER}"
    )
    assert m0_payload is not None, "m0 refit payload missing"
    for k, entry in enumerate(arms):
        wmap_payload = G._load_map_payload(
            G.Cfg(out_root=roots.gpu_out, stage_root=roots.stage_root, upload=False),
            f"wmap_{entry['arm_id']}_L{LAYER}",
        )
        assert wmap_payload is not None, entry["arm_id"]
        recompute_mapcols(
            roots,
            entry,
            m0_payload,
            wmap_payload,
            rb,
            dev,
            panel_anchor=not roots.smoke and entry["ctx_key"] != "bare",
        )
        print(f"[f1b] columns {k + 1}/{len(arms)} {entry['arm_id']} done", flush=True)

    _upload_f1b(roots)
    _atomic_json(
        roots.gpu_out / "f1b_done.json",
        {**_meta(), "n_fits": 1 + len(arms), "arms": [a["arm_id"] for a in arms]},
    )
    _phase("done")


def _load_rb(roots: Roots, arms: list[dict]) -> dict:
    if roots.smoke:  # data-level fixture (functions stay real); unit vectors
        rng = np.random.default_rng(FIT_SEED)
        out = {}
        for beh in sorted({a["beh_key"] for a in arms}):
            v = rng.standard_normal((LAYER + 2, SMOKE_FIT_DIM))
            out[beh] = v / np.linalg.norm(v, axis=1, keepdims=True)
        return out
    import issue1768_directions as D

    rb = D.load_rb_tensors(roots.i1768_root)
    assert set(rb) >= {"cas", "imp", "syc"}, sorted(rb)
    return rb


def _upload_f1b(roots: Roots) -> None:
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    _phase("f1b_upload")
    api = HfApi()
    for name in ("maps", "columns"):
        local = roots.gpu_out / name
        files = sorted(p for p in local.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
        assert files, (name, "nothing to upload")
        prefix = f"{roots.off_prefix}/{name}"
        hub._upload(
            local,
            repo_id=_data_repo(),
            repo_type="dataset",
            path_in_repo=prefix,
            raise_on_error=True,
        )
        expected = [f"{prefix}/{p.relative_to(local)}" for p in files]
        missing = hub.verify_repo_paths_uploaded(
            api, _data_repo(), expected, path_in_repo=prefix, repo_type="dataset"
        )
        assert not missing, f"f1b upload verify FAILED for {prefix}: missing {missing}"
        logger.info("[upload] %s verified (%d files)", prefix, len(files))


def _build_smoke_fixture(roots: Roots, arms: list[dict]) -> None:
    """Schema-exact tiny stores (REAL shas, D=32) so the REAL loaders + fit +
    recompute code run end-to-end on CPU. Fakes only the GPU-scale tensors."""
    import torch

    sample = X.load_corpus_sample(roots.i1768_root)
    sha_to_q = {r["sha"]: q for q, r in enumerate(sample["rows"])}
    subsets = {f: load_offfloor_subset(roots, f) for f in CONTENT_FAMILIES}
    scope = sorted(set(load_intersection(roots)))
    fam_rows = sorted(set(subsets["imp"]) | set(subsets["syc"]))
    union = set(J.load_subset(roots.config_dir)) | {x for f in subsets.values() for x in f}
    extra = [s for s in scope if s not in union][:SMOKE_EXTRA_ROWS]
    rows = [s for s in fam_rows + extra if s in sha_to_q]
    assert len(extra) > 4 * SMOKE_FIT_DIM, (len(extra), "smoke tr pool too small")
    rng = np.random.default_rng(FIT_SEED)

    def _store(path: Path, spans: tuple[str, ...]) -> None:
        if path.exists():
            return
        obj = {
            "arms": {
                sp: {
                    LAYER: torch.from_numpy(
                        rng.standard_normal((len(rows), SMOKE_FIT_DIM)).astype(np.float16)
                    )
                }
                for sp in spans
            },
            "row_sha": list(rows),
            "row_question_idx": [sha_to_q[s] for s in rows],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)

    _store(
        roots.i1768_root / "corpus_capture" / "base_content" / "pooled.pt", ("context", "response")
    )
    for a in arms:
        _store(
            roots.i1768_root / "corpus_capture" / a["arm_id"] / "pooled.pt", ("context", "response")
        )
        _store(roots.i1768_root / "corpus_capture_tf" / a["arm_id"] / "pooled_tf.pt", ("response",))
        anc_path = roots.gpu_out / "anchors" / f"{a['mix_arm_id']}.pt"
        if not anc_path.exists():
            anc_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "A_ctx": {LAYER: torch.from_numpy(rng.standard_normal(SMOKE_FIT_DIM))},
                    "A_ans": {LAYER: torch.from_numpy(rng.standard_normal(SMOKE_FIT_DIM))},
                },
                anc_path,
            )
    logger.info("[f1b-smoke] fixture stores written: %d rows, D=%d", len(rows), SMOKE_FIT_DIM)


# ── F3: parquet rewrite + production race/followup entrypoints ───────────────


def rewrite_arm_parquet(roots: Roots, entry: dict, subset: set[str]) -> Path:
    """in_judge_subset := family subset; MAP_COLS := F1b refit values (parent
    values preserved as *_parentmap record-only columns; plan section 6 line 2)."""
    import pandas as pd

    arm_id = entry["arm_id"]
    out = roots.p1_root / "predictor_tables" / f"{arm_id}_L{LAYER}.parquet"
    tab = pd.read_parquet(_stage_parent_parquet(roots, arm_id))
    tab["in_judge_subset"] = tab["sha"].isin(subset)
    n_sub = int(tab["in_judge_subset"].sum())
    assert n_sub > 0, (arm_id, "empty off-floor subset in the parquet")
    mapcols_path = roots.columns_dir / f"{arm_id}_L{LAYER}_mapcols.parquet"
    if not mapcols_path.exists() and not roots.smoke:
        from explore_persona_space.orchestrate import hub

        hub.stage_hub_file(
            _data_repo(),
            f"{OFF_HF_PREFIX}/columns/{mapcols_path.name}",
            mapcols_path,
            repo_type="dataset",
        )
    mc = pd.read_parquet(mapcols_path).set_index("sha")
    # r1 Critical 1: the #1768 stores carry 18 dup-sha rows (100 rows) and
    # `load_corpus_cell` keeps them, so a production mapcols parquet is
    # dup-sha; `Series.map` on a non-unique-index Series raises pandas
    # InvalidIndexError. keep="first" is VALUE-SAFE: dup shas are never
    # subset members (excluded from the candidate pool; asserted absent from
    # the parent subset in `_assumption1_probe`), so only never-raced
    # non-subset rows can read a first-occurrence value.
    mc = mc[~mc.index.duplicated(keep="first")]
    covered = tab.loc[tab["in_judge_subset"], "sha"].isin(mc.index)
    assert covered.all(), (arm_id, int((~covered).sum()), "mapcols missing subset shas")
    for col in MAP_COLS:
        tab[f"{col}_parentmap"] = tab[col]
        joined = tab["sha"].map(mc[col])
        tab[col] = joined.where(joined.notna(), tab[col])
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out, index=False)
    return out


def _stage_validation(roots: Roots) -> None:
    from explore_persona_space.orchestrate import hub

    local = roots.p1_root / "validation" / "tf_margin_arm.jsonl"
    if roots.smoke or local.exists():
        return  # smoke: validation_read writes its documented skip payload
    hub.stage_hub_file(
        _data_repo(), f"{HF_PREFIX}/validation/tf_margin_arm.jsonl", local, repo_type="dataset"
    )


def _run(cmd: list[str], what: str) -> None:
    print(f"[f3] exec {what}: {' '.join(cmd[:8])} ...", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ}, check=False)
    assert proc.returncode == 0, (what, proc.returncode)


def phase_f3(roots: Roots, b_draws: int, n_perm: int) -> None:
    _phase("f3_rewrite")
    arms = content_arm_entries(roots)
    if roots.smoke:
        arms = [a for a in arms if a["arm_id"] in SMOKE_ARMS]
    subsets = {f: set(load_offfloor_subset(roots, f)) for f in CONTENT_FAMILIES}
    for k, entry in enumerate(arms):
        rewrite_arm_parquet(roots, entry, subsets[entry["beh_key"]])
        print(f"[f3] parquet {k + 1}/{len(arms)} {entry['arm_id']} rewritten", flush=True)
    _stage_validation(roots)
    # stage cas-arm judge_inputs from the parent HF prefix (robustness reads);
    # base_content + imp/syc shards were built by F0 in the offfloor inputs dir.
    cas_units = [a["arm_id"] for a in arms if a["beh_key"] == "cas"]
    if cas_units:
        J.stage_judge_inputs(roots.inputs_dir, cas_units)

    _phase("f3_race")
    arm_ids = ",".join(a["arm_id"] for a in arms)
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/issue1900_race.py",
            "--config-dir",
            str(roots.config_dir),
            "--p1-root",
            str(roots.p1_root),
            "--judge-dir",
            str(roots.judge_out),
            "--inputs-dir",
            str(roots.inputs_dir),
            "--out-dir",
            str(roots.race_out),
            "--arms",
            arm_ids,
            "--b-draws",
            str(b_draws),
            "--n-perm",
            str(n_perm),
        ],
        "race",
    )
    _phase("f3_followup")
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/issue1900_followup_free.py",
            "--config-dir",
            str(roots.config_dir),
            "--p1-root",
            str(roots.p1_root),
            "--judge-dir",
            str(roots.judge_out),
            "--race-dir",
            str(roots.race_out),
            "--out-dir",
            str(roots.race_out / "followup_free"),
            "--fig-dir",
            str(roots.fig_dir),
            "--b-draws",
            str(b_draws),
            "--arms",
            arm_ids,
            "--skip-figs",
        ],
        "followup_free",
    )
    print("[f3] done", flush=True)


# ── figures ──────────────────────────────────────────────────────────────────


def phase_figs(roots: Roots) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import issue1900_figs as FG
    import issue1900_race as R

    _phase("figs")
    roots.fig_dir.mkdir(parents=True, exist_ok=True)
    payloads = FG._load_arm_jsons(roots.race_out)
    payloads = [p for p in payloads if p.get("kind") == "content"]
    made: list[Path] = []
    p = FG._heatmap(
        payloads,
        "dv_level",
        "offfloor_race_content_level",
        roots.fig_dir,
        "Off-floor race (graded LEVEL) — content arms (base score > 0)",
    )
    if p:
        made.append(p)
    resid = [
        json.loads(f.read_text())
        for f in sorted((roots.race_out / "followup_free").glob("arm_*.json"))
    ]
    p = FG._heatmap(
        resid,
        "dv_resid",
        "offfloor_residualized_race_content",
        roots.fig_dir,
        "Off-floor propensity-residualized change race — content arms",
    )
    if p:
        made.append(p)

    # parent-vs-offfloor paired dots (per-arm rho; descriptive only, plan section 6)
    cands = ("p7", "p1", "p2")
    fig, axes = plt.subplots(1, len(cands), figsize=(11, 3.6), layout="constrained")
    for ax, cand in zip(axes, cands):
        xs, ys = [], []
        for pay in payloads:
            parent_path = roots.parent_race_dir / f"arm_{pay['arm_id']}.json"
            if not parent_path.exists():
                continue
            par = json.loads(parent_path.read_text())
            xs.append(par["observed_rho"]["dv_level"].get(cand))
            ys.append(pay["observed_rho"]["dv_level"].get(cand))
        ax.scatter(xs, ys, s=28, color="#0072B2")
        lim = [min(xs + ys + [0]) - 0.05, max(xs + ys + [0]) + 0.05]
        ax.plot(lim, lim, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlabel("parent run rho (full subset)")
        ax.set_ylabel("off-floor rho")
        ax.set_title(FG.CANDIDATE_NAMES.get(cand, cand))
    fig.suptitle("Per-arm Spearman rho, parent vs off-floor (graded LEVEL; descriptive)")
    made.append(FG._save(fig, "offfloor_parent_vs_offfloor_rho", roots.fig_dir))

    # per-arm DV-vs-candidate scatters (points = contexts), one figure per candidate
    arms = content_arm_entries(roots)
    if roots.smoke:
        arms = [a for a in arms if a["arm_id"] in SMOKE_ARMS]
    asms = [
        R.assemble_content_arm(a, roots.p1_root / "predictor_tables", roots.judge_out) for a in arms
    ]
    for cand in cands:
        ncol = 4 if len(asms) > 4 else len(asms)
        nrow = int(np.ceil(len(asms) / ncol))
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow), layout="constrained", squeeze=False
        )
        for i, asm in enumerate(asms):
            ax = axes[i // ncol][i % ncol]
            f = asm["frame"]
            xv = f["p7"] if cand == "p7" else f[R.CANDIDATE_COLS[cand]]
            ax.scatter(xv, f["dv_level"], s=4, alpha=0.25, color="#0072B2", rasterized=True)
            ax.set_title(FG.arm_plain(asm["arm"]["arm_id"]), fontsize=7)
        for j in range(len(asms), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"Off-floor contexts: graded leakage vs {FG.CANDIDATE_NAMES.get(cand, cand)}")
        made.append(FG._save(fig, f"offfloor_scatter_dv_vs_{cand}", roots.fig_dir))
    for pth in made:
        print(f"[figs] wrote {pth}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase", required=True, choices=("f0", "f1a", "f0b", "f1b", "f2", "f3", "figs", "smoke")
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="scratch roots + tiny slices (never canonical eval_results)",
    )
    ap.add_argument("--dry-run", action="store_true", help="judge phases: 0 API calls")
    ap.add_argument("--device", default="cuda", help="f1b fit device (smoke: cpu)")
    ap.add_argument("--b-draws", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    args = ap.parse_args()

    smoke = args.smoke or args.phase == "smoke"
    roots = make_roots(smoke)
    for d in (
        roots.data,
        roots.evalr,
        roots.cfg_out,
        roots.judge_out,
        roots.race_out,
        roots.inputs_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    b_draws = args.b_draws or (200 if smoke else R_B_BOOT())
    n_perm = args.n_perm or (100 if smoke else R_N_PERM())

    if args.phase == "smoke":
        # chained scratch end-to-end: the SAME phase functions, tiny slices,
        # cached-judge subsets (parent scores), CPU fixture fits (D=32).
        phase_f0(roots)
        phase_f1a(roots, dry_run=True)
        # cached-judge selection record: parent base rows only (no live calls)
        for fam in CONTENT_FAMILIES:
            parent = _parent_scores(roots, f"base_{fam}")
            merged = _merged_payload(
                parent,
                None,
                {r["sha"] for r in parent["rows"]},
                f"selection_base_{fam}",
                "SMOKE cached-judge selection",
            )
            _atomic_json(roots.judge_out / f"selection_base_{fam}.json", merged)
        phase_f0b(roots)
        phase_f2(roots, dry_run=False)  # smoke branch: dry-run items + cached copies
        phase_f1b(roots, device="cpu")
        phase_f3(roots, b_draws, n_perm)
        phase_figs(roots)
        _phase("done")
        sys.stdout.flush()
        sys.exit(0)

    if args.phase == "f0":
        phase_f0(roots)
    elif args.phase == "f1a":
        phase_f1a(roots, dry_run=args.dry_run)
    elif args.phase == "f0b":
        phase_f0b(roots)
    elif args.phase == "f1b":
        phase_f1b(roots, device=("cpu" if smoke else args.device))
    elif args.phase == "f2":
        phase_f2(roots, dry_run=args.dry_run)
    elif args.phase == "f3":
        phase_f3(roots, b_draws, n_perm)
    elif args.phase == "figs":
        phase_figs(roots)
    _phase("done")
    sys.stdout.flush()
    sys.exit(0)


def R_B_BOOT() -> int:
    import issue1900_race as R

    return R.B_BOOT


def R_N_PERM() -> int:
    import issue1900_race as R

    return R.N_PERM


if __name__ == "__main__":
    main()
