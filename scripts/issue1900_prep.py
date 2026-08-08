"""#1900 P0 prep — judge-subset draw + pinned arm config (VM-side; plan §4 P0).

Builds, from the #1768 banked inputs (pin `c0726728…`):

- ``data/issue_1900/config/subset.json`` — the 4,000-sha judge subset, drawn
  stratified by (corpus source x prompt-length quintile) at seed 1900 from
  the TRAIN rows only (valtest carries 82 internal duplicate shas — drawing
  from train guarantees subset ∩ refit-row shas = ∅ at sha grain, plan §12.17),
  restricted to shas present in ALL selected units' kept-row sets (global
  intersection asserted ≥ 0.9 x 16,400 — the #1768 loader's join-floor
  convention; kill criterion 2 reported per arm).
- ``data/issue_1900/config/arms.json`` — the 18 selected arms (plan §4 pick
  criteria) with per-arm adapter / full-FT checkpoint HF paths resolved via
  the #1768 mechanism (`issue1768_cells.adapter_subfolder` /
  `ft_ckpt_subfolder`), the FT→LoRA mix mapping (`delta_arm_for` — the
  verdict manifest's `mix_pos_sources` is NULL for the 4 FT arm ids), and the
  verdict manifest's RESOLVED revision at read time.

Then UPLOADS the config plus the two P1c pool inputs
(``eval_results/issue_722/tf_margin/margin_chain.json`` +
``eval_results/issue_661/judge_filter.json``) to
``issue1900_leakrace/config/`` on the HF data repo — the CANONICAL read path
for P1/P1c on EVERY lane (`data/*` is gitignored, so a branch copy alone
would silently no-op on clone-based lanes — plan §4 P0.3).

The kept-row sha sets come from each unit's small ``rows_spans.json`` (the
per-unit span sidecar `issue1768_capture.run_corpus_unit` writes) — never the
706 MB ``pooled.pt`` stores; the tf trees teacher-force the BASE tree's kept
rows verbatim (`run_corpus_tf_unit`), so they add no constraint.

Intersection asserts fail loud; prompt TEXT is never printed (LMSYS/WildChat
real-user rows — digest-only handling).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before numpy/hub: shared-VM thread caps + HF credentials

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1900.prep")

ISSUE = 1900
HF_PREFIX = "issue1900_leakrace"
CONFIG_HF_PREFIX = f"{HF_PREFIX}/config"
# Plan §10 pin: the #1768 corpus_capture / delta_tf / inputs tree revision.
CORPUS_PIN = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"
SUBSET_SEED = 1900
N_SUBSET = 4_000
N_QUINTILES = 5
JOIN_FLOOR = 0.9  # the #1768 loader convention (issue1768_fit.load_corpus_cell)
MAX_ARMS_BELOW_FLOOR = 6  # plan §7 kill criterion 2
# Plan §4 P0: corpus-source proportions fixed at the realized #1768 train split.
CORPUS_PROPORTIONS = {"lmsys": 8_211, "wildchat": 6_789}
MARGIN_CHAIN_SHA256_PREFIX = "2cb680f3"  # plan §10 (consistency-check 2026-07-30)

# Plan §4 arm picks (18; criteria: span behaviors x training contexts x
# regimes x methods, prefer seed 42, one seed pair, exclude content
# demonstration/icl arms — lr-heterogeneous + out-of-band caveat per #1768).
CONTENT_ARMS = (
    "cas-pers-con-lr1e5-s42",
    "cas-pers-po-lr1e5-s42",
    "cas-bare-con-lr1e5-s42",
    "cas-pers-ft-con-s42",
    "imp-pers-con-lr3e5-s42",
    "imp-pers-con-lr3e5-s137",  # seed pair
    "imp-pers-po-lr1e5-s42",
    "imp-pers-ft-con-s42",
    "syc-bare-con-lr1e5-s42",
    "syc-conv-con-lr1e5-s42",
    "syc-pers-po-lr1e5-s42",
    "syc-pers-ft-con-s42",
)
MARKER_ARMS = (
    "mk-pers-con-lr5e6-s42",
    "mk-bare-con-lr5e6-s42",
    "mk-conv-con-lr5e6-s42",
    "mk-icl-con-lr5e6-s42",
    "mk-pers-po-lr5e6-s42",
    "mk-pers-ft-con-s42",
)
SELECTED_ARMS = CONTENT_ARMS + MARKER_ARMS
PRIMARY_LAYER = {"content": 19, "marker": 25}  # plan §11 pre-registered


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — metadata only
        return "unknown"


def _meta() -> dict:
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "numpy": np.__version__,
        "issue": ISSUE,
        "corpus_pin": CORPUS_PIN,
        "subset_seed": SUBSET_SEED,
    }


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _main_repo_root() -> Path:
    """Parent of the git COMMON dir — worktree-safe main-checkout root."""
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent


def _resolve_manifest(hf_dl: Path) -> tuple[dict, str, str]:
    """(manifest, resolved_revision_sha, sha256) — the HF mirror at main's tip.

    Also seeds `issue1768_cells._load_manifest`'s module-side fallback cache so
    every `X.all_arms()` call in this process reads EXACTLY the recorded bytes
    (the sparse worktree carries no eval_results/issue_1481 copy).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    refs = hub.retry_transient(
        lambda: HfApi().list_repo_refs(X.HF_DATA_REPO, repo_type="dataset"),
        what="data-repo refs (manifest revision resolve)",
    )
    main_sha = next(b.target_commit for b in refs.branches if b.name == "main")
    local = hf_dl / "verdict_manifest.json"
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{X.HF_PREFIX}/inputs/verdict_manifest.json",
        local,
        repo_type="dataset",
        revision=main_sha,
        overwrite=True,
    )
    raw = local.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    committed = X.VERDICT_MANIFEST
    if committed.exists():
        committed_sha = hashlib.sha256(committed.read_bytes()).hexdigest()
        assert committed_sha == sha256, (
            "verdict manifest HF mirror != committed eval_results copy: "
            f"mirror {sha256} vs committed {committed_sha} — refuse to pin a divergent identity"
        )
    else:
        # Seed the module fallback cache (X._load_manifest) with the pinned bytes.
        cache = SCRIPTS_DIR / ".i1768_verdict_manifest.json"
        cache.write_bytes(raw)
    return json.loads(raw.decode("utf-8")), main_sha, sha256


def _stage_pinned(hf_dl: Path, repo_path: str, name: str) -> Path:
    from explore_persona_space.orchestrate import hub

    local = hf_dl / name
    if not local.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO, repo_path, local, repo_type="dataset", revision=CORPUS_PIN
        )
    return local


def _unit_kept_shas(hf_dl: Path, unit_id: str) -> list[str]:
    """Kept-row shas of one corpus_capture unit (rows_spans.json sidecar @ pin)."""
    local = _stage_pinned(
        hf_dl / "rows_spans",
        f"{X.HF_PREFIX}/corpus_capture/{unit_id}/rows_spans.json",
        f"{unit_id}.json",
    )
    rows = json.loads(local.read_text())["rows"]
    shas = [r["prompt_sha"] for r in rows]
    return shas


def _sha_dedup(sample: dict) -> tuple[list[dict], dict[str, int]]:
    """Train rows (sha-unique, asserted) + realized per-corpus counts."""
    n_train = sample["n_train"]
    train_rows = sample["rows"][:n_train]
    shas = [r["sha"] for r in train_rows]
    assert len(set(shas)) == n_train, (
        f"train sha dedup violated upstream: {len(set(shas))} unique of {n_train}"
    )
    valtest_shas = {r["sha"] for r in sample["rows"][n_train:]}
    assert not (set(shas) & valtest_shas), "train ∩ valtest sha overlap (upstream invariant)"
    counts: dict[str, int] = {}
    for r in train_rows:
        counts[r["corpus"]] = counts.get(r["corpus"], 0) + 1
    return train_rows, counts


def _largest_remainder(total: int, weights: list[float]) -> list[int]:
    raw = [total * w / sum(weights) for w in weights]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def draw_subset(
    train_rows: list[dict], eligible_shas: set[str], corpus_counts: dict[str, int]
) -> tuple[list[str], dict]:
    """Seeded stratified draw: (corpus source x prompt-length quintile)."""
    assert corpus_counts == CORPUS_PROPORTIONS, (
        f"realized train corpus counts {corpus_counts} != plan-fixed {CORPUS_PROPORTIONS}"
    )
    rng = np.random.default_rng(SUBSET_SEED)
    corpora = sorted(CORPUS_PROPORTIONS)
    quotas = dict(
        zip(corpora, _largest_remainder(N_SUBSET, [CORPUS_PROPORTIONS[c] for c in corpora]))
    )
    chosen: list[str] = []
    strata_report: dict[str, dict] = {}
    for corpus in corpora:
        rows = [r for r in train_rows if r["corpus"] == corpus and r["sha"] in eligible_shas]
        assert len(rows) >= quotas[corpus], (corpus, len(rows), quotas[corpus])
        lengths = np.asarray([len(r["prompt"]) for r in rows], dtype=np.float64)
        edges = np.quantile(lengths, np.linspace(0, 1, N_QUINTILES + 1)[1:-1])
        strata = np.searchsorted(edges, lengths, side="right")  # 0..4
        sizes = [int((strata == q).sum()) for q in range(N_QUINTILES)]
        q_quotas = _largest_remainder(quotas[corpus], [max(s, 0) for s in sizes])
        picked_per_q = []
        for q in range(N_QUINTILES):
            pool = sorted(rows[i]["sha"] for i in np.where(strata == q)[0])
            assert len(pool) >= q_quotas[q], (corpus, q, len(pool), q_quotas[q])
            picked = rng.choice(np.asarray(pool, dtype=object), size=q_quotas[q], replace=False)
            chosen.extend(str(s) for s in picked)
            picked_per_q.append(q_quotas[q])
        strata_report[corpus] = {
            "quota": quotas[corpus],
            "n_eligible": len(rows),
            "length_quintile_edges_chars": [float(e) for e in edges],
            "stratum_sizes": sizes,
            "stratum_quotas": picked_per_q,
        }
    assert len(chosen) == N_SUBSET, len(chosen)
    assert len(set(chosen)) == N_SUBSET, "duplicate shas in the judge subset"
    return sorted(chosen), strata_report


def build_arms(manifest_rev: str, manifest_sha256: str, registry: dict) -> dict:
    """arms.json payload: 18 selected arms + resolution provenance (plan §4 P0.3)."""
    index = {a.arm_id: a for a in X.all_arms()}
    missing = [a for a in SELECTED_ARMS if a not in index]
    assert not missing, f"selected arms absent from the #1768 fleet registry: {missing}"
    mix_sources = registry["mix_pos_sources"]
    arms = []
    for arm_id in SELECTED_ARMS:
        arm = index[arm_id]
        mix_arm_id = X.delta_arm_for(arm)  # FT arms map to the matched pers-LoRA cell's mix
        src = mix_sources.get(mix_arm_id)
        assert src and src.get("pos_path"), (
            f"{arm_id}: mapped LoRA mix {mix_arm_id} has no pos source in arm_registry "
            f"(plan §4 P0 FT→LoRA mapping assert)"
        )
        entry = {
            "arm_id": arm_id,
            "kind": arm.kind,
            "beh_key": arm.beh_key,
            "ctx_key": arm.ctx_key,
            "regime": arm.regime,
            "seed": arm.seed,
            "lr": arm.lr,
            "step": arm.step,
            "method": arm.method,
            "selection_read": arm.selection_read,
            "base_unit": X.base_unit_for(arm_id),
            "primary_layer": PRIMARY_LAYER[arm.kind],
            "mix_arm_id": mix_arm_id,
            "mix_pos_path": src["pos_path"],
            "mix_layout": src["layout"],
        }
        if arm.method == "lora":
            entry["adapter_repo"] = X.HF_MODEL_REPO
            entry["adapter_subfolder"] = X.adapter_subfolder(arm)
        else:
            entry["ft_repo"] = X.FT_OVERFLOW_REPO
            entry["ft_subfolder"] = X.ft_ckpt_subfolder(arm)
        if arm.kind == "marker":
            # #1481 verdict window value for the P1a adapter-application smoke
            # (assert median Δ logP within ±1 nat of this — plan §4 P1a).
            entry["manifest_delta_logp_mean"] = arm.selection_read
        arms.append(entry)
    n_ft = sum(1 for a in arms if a["method"] == "ft")
    assert len(arms) == 18 and n_ft == 4, (len(arms), n_ft)
    return {
        "arms": arms,
        "pick_criteria": (
            "span behaviors x training contexts x regimes x methods; prefer seed 42; "
            "one seed pair (imp-pers-con-lr3e5 s42/s137) for a replication read; content "
            "demonstration/icl arms excluded (lr-heterogeneous + out-of-band per #1768); "
            "po arms included as the high-leakage bets (contrastive-negatives rule #18/#207)"
        ),
        "verdict_manifest": {
            "hf_path": f"{X.HF_PREFIX}/inputs/verdict_manifest.json",
            "resolved_revision": manifest_rev,
            "sha256": manifest_sha256,
        },
        "corpus_pin": CORPUS_PIN,
        "ft_mix_mapping_note": (
            "arm_registry.mix_pos_sources is keyed by LoRA arm id (NULL for ft arms); "
            "ft arms carry mix_arm_id = the matched pers-LoRA cell (issue1768_cells."
            "delta_arm_for) and P1b passes the MAPPED LoRA arm into _mix_positive_rows"
        ),
        **_meta(),
    }


def upload_config(config_dir: Path, expected: list[str]) -> None:
    """One folder commit to the canonical HF mirror + exact-set verify."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    url = hub._upload(
        config_dir,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=CONFIG_HF_PREFIX,
        raise_on_error=True,
    )
    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        X.HF_DATA_REPO,
        [f"{CONFIG_HF_PREFIX}/{name}" for name in expected],
        path_in_repo=CONFIG_HF_PREFIX,
        repo_type="dataset",
    )
    assert not missing, f"config mirror verify failed — missing on Hub: {missing}"
    logger.info("[p0] config mirror verified at %s (%s)", CONFIG_HF_PREFIX, url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "data/issue_1900/config",
        help="output dir for subset.json/arms.json (worktree branch copy)",
    )
    ap.add_argument(
        "--hf-dl",
        type=Path,
        default=REPO_ROOT / "data/issue_1900/hf_dl",
        help="staging cache for pinned #1768 inputs (re-downloadable)",
    )
    ap.add_argument("--no-upload", action="store_true", help="skip the HF config mirror upload")
    args = ap.parse_args()

    main_root = _main_repo_root()
    margin_chain = main_root / "eval_results/issue_722/tf_margin/margin_chain.json"
    judge_filter = main_root / "eval_results/issue_661/judge_filter.json"
    for p in (margin_chain, judge_filter):
        assert p.exists(), f"P1c pool input missing on the VM: {p}"
    mc_sha = hashlib.sha256(margin_chain.read_bytes()).hexdigest()
    assert mc_sha.startswith(MARGIN_CHAIN_SHA256_PREFIX), (
        f"margin_chain.json sha {mc_sha} != plan §10 pin prefix {MARGIN_CHAIN_SHA256_PREFIX}"
    )

    manifest, manifest_rev, manifest_sha = _resolve_manifest(args.hf_dl)
    del manifest  # identity recorded; X.all_arms() reads the seeded cache
    registry_path = _stage_pinned(
        args.hf_dl, f"{X.HF_PREFIX}/arm_registry.json", "arm_registry.json"
    )
    registry = json.loads(registry_path.read_text())
    arms_payload = build_arms(manifest_rev, manifest_sha, registry)

    sample_path = _stage_pinned(
        args.hf_dl, f"{X.HF_PREFIX}/inputs/corpus_sample.json", "corpus_sample.json"
    )
    sample = json.loads(sample_path.read_text())
    n_total = len(sample["rows"])
    assert n_total == sample["n_train"] + sample["n_val"] + sample["n_test"], n_total
    train_rows, corpus_counts = _sha_dedup(sample)

    # Global kept-row intersection over base units + all 18 arms (tf trees
    # teacher-force the base tree's rows — no extra constraint, recorded).
    units = list(X.BASE_UNITS) + list(SELECTED_ARMS)
    kept: dict[str, set[str]] = {}
    for u in units:
        shas = _unit_kept_shas(args.hf_dl, u)
        kept[u] = set(shas)
        logger.info("[p0] %s kept rows: %d (unique %d)", u, len(shas), len(kept[u]))
    inter = set.intersection(*kept.values())
    floor = JOIN_FLOOR * n_total
    per_arm_join = {}
    below = []
    for arm_id in SELECTED_ARMS:
        base = kept[X.base_unit_for(arm_id)]
        frac = len(base & kept[arm_id]) / n_total
        per_arm_join[arm_id] = round(frac, 6)
        if frac < JOIN_FLOOR:
            below.append(arm_id)
    assert len(inter) >= floor, (
        f"global sha intersection {len(inter)} < {floor:.0f} (0.9 x {n_total}) — "
        f"stores unusable at the required grain (plan §7 kill criterion 2)"
    )
    assert len(below) <= MAX_ARMS_BELOW_FLOOR, (
        f"{len(below)} arms below the {JOIN_FLOOR} join floor (> {MAX_ARMS_BELOW_FLOOR}): {below}"
    )

    subset, strata_report = draw_subset(train_rows, inter, corpus_counts)
    subset_set = set(subset)
    other_shas = {r["sha"] for r in sample["rows"] if r["sha"] not in subset_set}
    assert not (subset_set & other_shas), "subset ∩ refit-row shas non-empty (plan §12.17)"

    args.config_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.config_dir / "subset.json",
        {
            "shas": subset,
            "n": len(subset),
            "drawn_from": "train rows only (valtest carries 82 internal dup shas; §12.17)",
            "strata": strata_report,
            "intersection_size": len(inter),
            "n_corpus_rows": n_total,
            "per_arm_join_fraction": per_arm_join,
            "arms_below_join_floor": below,
            "units_intersected": units,
            "tf_tree_note": "corpus_capture_tf teacher-forces the base tree's kept rows "
            "(issue1768_capture.run_corpus_tf_unit) — no extra sha constraint",
            **_meta(),
        },
    )
    _atomic_json(args.config_dir / "arms.json", arms_payload)
    (args.config_dir / "margin_chain.json").write_bytes(margin_chain.read_bytes())
    (args.config_dir / "judge_filter.json").write_bytes(judge_filter.read_bytes())
    logger.info(
        "[p0] config written: %s (subset n=%d, intersection=%d, arms=%d, margin_chain sha=%s)",
        args.config_dir,
        len(subset),
        len(inter),
        len(arms_payload["arms"]),
        mc_sha[:16],
    )

    if not args.no_upload:
        upload_config(
            args.config_dir,
            ["subset.json", "arms.json", "margin_chain.json", "judge_filter.json"],
        )
    print(f"[phase=p0_done] subset={len(subset)} intersection={len(inter)} arms=18", flush=True)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
