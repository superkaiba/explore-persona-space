"""#1947 Result 3 — condition-grain theory battery over the 52 content arms (inline round).

Runs the leakage-theory paper's OWN prescribed tests (assumptions
``a:source-write`` / ``a:rank-one-gated-write`` / ``a:bilinear-gate``,
~/overleaf-6a2df2d2/main.tex) at the paper's grain — context CONDITIONS —
on all 52 #1947 content arms (48 single-visit + 4 repeat-regime), in BOTH
response trees:

- **on-policy (PRIMARY)**: v_theta(C') = profile of the model's OWN greedy
  answers over the #1979 condition panel (50 prefixes x 60 shared queries);
  base and trained models each generate their own answers.
- **matched-text (SECONDARY)**: trained model teacher-forced on the base
  model's answers — the weights-carried share of the update.

Reuses ``issue1979_gpu``'s production unit functions verbatim (generation +
capture ``run_f1a``, matched-text TF ``run_f1b_writes``, training-anchor
delta capture ``run_f1c``) with three DOCUMENTED module overrides applied at
import (upload prefix; max_new 2048 — #1979's 1024 cap realized 5.7-8.5%
cap-hit in EVERY panel family, breaching the >2% re-gen trigger, so both
trees are regenerated at 2048; anchors keep ALL mix positives instead of a
20-row subsample). Config is built VM-side (``--build-config``) from the
committed #1979 panel/queries (verbatim, sha-pinned) + a NEW 52-arm
``arms.json`` + 2 SOURCE-ONLY panel extension members (icl_prefix_impolite /
icl_prefix_writing_style — the behavior-specific ICL training contexts the
#1979 panel lacks; they serve only as each icl arm's source condition C and
never enter any arm's target set).

Battery blocks (per arm x tree x layer 14/19/25):

- **A (rank-one gated write)**: w_hat = Delta_v(C); realized gate
  g_hat(C'_i) = w_hat.T Delta_v(C'_i) / |w_hat|^2; scalarity residual
  distribution; Delta_V spectrum (top-1 var share UNCENTERED — the paper's
  own statistic — plus centered), sigma2/sigma1, cos(u1, w_hat); the
  low-rank fallback with QUERY-ALIGNED split-half cross-validated
  reconstruction error over m=1..10 (one query partition applied to every
  condition — #763; directions cross-validated, never held-out conditions).
- **B (source write)**: cos(w_hat, delta), eta_hat, relative residual,
  cos(w_hat, r_B); isotropic + corpus-covariance norm-matched nulls.
- **C (bilinear gate)**: predicted vs realized gate for the paper's six
  named combos k in {c_C, psi(t), c_C+psi(delta)} x M in {I, (Sigma_c+lI)^-1}
  (psi = identity: keys and queries share the layer's residual space; the
  paper's 4th ablation key psi(delta) is reported as a supplement), q pooled
  as context span-mean AND last-prompt-token (last-prompt runs M=I only —
  no matched-pooling Sigma bank exists); permutation null over conditions.
- **D (operator framing)**: map fits SKIPPED (n_train < d at both grains —
  reported); data-space read with the mean shift separated from the
  centered remainder.
- **Nulls**: every rank statistic vs an n-matched null (iid gaussian
  columns scaled to the realized per-column norms) at BOTH grains
  (condition n<=50; prompt n<=3000).

CLI::

    uv run python scripts/issue1947_result3_theory_battery.py --build-config
    uv run python scripts/issue1947_result3_theory_battery.py --smoke-then-full \
        --out-root <root>            # pod-side: tiny smoke leg, then full leg
    uv run python scripts/issue1947_result3_theory_battery.py --harvest --figures
    uv run python scripts/issue1947_result3_theory_battery.py --import-check
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before any tokenizer/torch import: thread caps + HF credentials

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1979_gpu as G  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i1947_r3")

ISSUE = 1947
HF_PREFIX = "issue1947_result3theory"
LAYERS = tuple(G.LAYERS_1979)  # (14, 19, 25)
PRIMARY_LAYER = 19
SEED = 1947
PILOT_ARM = "syc-pers-con-sv-s42"
SMOKE_ARMS = ("syc-pers-con-sv-s42", "imp-pers-con-rep-s42", "imp-icl-con-sv-s42")
MAX_NEW_R3 = 2048  # brief cap; #1979's 1024 breached the >2% cap-hit trigger in every family
MAX_MODEL_LEN_R3 = 6144  # 3072 F0 prompt budget + 2048 decode + headroom
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
MIX_BUCKET = "issue1947_singlevisit/mixes"
# sha256 pins of the VERBATIM #1979 config files this round reuses are recorded in
# config_manifest.json at --build-config and asserted at every pod-side load.
EXT_SOURCE_CIDS = ("icl_prefix_impolite", "icl_prefix_writing_style")
EXT_FAMILY = "trained-ext"  # source-only extension members; never targets
SRC_CONDITION_BY_CTX = {
    "pers": "persona_software_engineer",
    "bare": "bare",
    "conv": "wildchat_prefix_real545",
}  # icl resolves per behavior: icl_prefix_<behavior>
# v2 battery spec (post-dispatch teammate directives, 2026-08-03):
DUP_TARGET_DROP = ("neg_default_assistant",)  # render-identical to `bare` (sha 83840f5e82);
#   kept in capture, DROPPED from every target set so the duplicate condition never
#   double-counts in Delta_V spectra / gate races.
M0_HUB_PATH = "issue1900_leakrace/maps/m0_L19.pt"  # the banked base map (pinned @ I1900_PIN)
PSI_LAYER = 19  # psi + map-residual live at L19 only (the banked M0's layer)
PSI_LAMBDAS = tuple(float(x) for x in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4))
LADDER_KS = (128, 256, 512)  # PCA input reduction sweep (n<d discipline, #1701)
N_NULL_COND = 200  # n-matched null draws, condition grain
N_NULL_PROMPT = 50  # n-matched null draws, prompt grain (L19 only)
N_PERM_GATE = 2000  # gate permutation null draws
CV_M_MAX = 10
CV_ELBOW_EPS = 0.01  # headline m* = smallest m with incremental CV-error improvement < eps
QUERY_FLOOR_FRAC = 2.0 / 3.0  # per-condition realized-query floor (fraction of sliced queries)
MAX_HEAVY_CONCURRENT = 8  # 8 coexisting merged dirs (8-GPU saturation; disk sized for it)
PHASE_HEADROOM_GB = {"gen": 150.0, "mt": 40.0, "anchor": 10.0, "battery": 20.0}
WORKER_HEADROOM_GB = 5.0
FAILURE_BUDGET = 5
SYSTEMIC_EXC_REPEAT = 3

# ── module overrides on the reused #1979 driver (documented; process-wide: the
#    dispatcher AND every worker subprocess run THIS script, so they apply
#    consistently in every process that calls G's unit functions) ─────────────
G.HF_PREFIX_1979 = HF_PREFIX  # upload destinations inside run_f1a/f1b/f1c
X.MAX_NEW_CONTENT = MAX_NEW_R3
X.MAX_MODEL_LEN = MAX_MODEL_LEN_R3
G.N_ANCHOR_ROWS = 10**9  # delta anchors keep ALL mix positives (300 sv / 20 rep)

CONFIG_FILES_R3 = ("prefix_panel.json", "queries.json", "arms.json", "ext_members.json")
EVAL_DIR = REPO_ROOT / "eval_results" / "issue_1947" / "result3_theory"
FIG_DIR = REPO_ROOT / "figures" / "issue_1947" / "result3_theory"
I1979_CONFIG = REPO_ROOT / "eval_results" / "issue_1979" / "config"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _meta() -> dict:
    return G._meta() | {"issue": ISSUE, "hf_prefix": HF_PREFIX, "r3_seed": SEED}


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ── Cfg ───────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Cfg:
    out_root: Path
    config_dir: Path
    panel_limit: int | None = None
    query_limit: int | None = None
    arms_filter: tuple[str, ...] = ()
    skip_upload: bool = False
    tf_batch: int = X.TF_BATCH_SIZE
    max_parallel: int | None = None
    smoke: bool = False
    null_cond: int = N_NULL_COND
    null_prompt: int = N_NULL_PROMPT
    perm_gate: int = N_PERM_GATE
    terminal_token: str = "done"  # pilot leg passes "pilot_done" (reserved-token discipline)

    @property
    def limited(self) -> bool:
        return self.panel_limit is not None or self.query_limit is not None

    def gcfg(self) -> "G.Cfg":
        """A real G.Cfg so the reused #1979 unit functions run verbatim."""
        return G.Cfg(
            out_root=self.out_root,
            config_dir=self.config_dir,
            phases=(),
            skip_upload=self.skip_upload,
            tf_batch=self.tf_batch,
        )

    def worker_flags(self) -> list[str]:
        flags = ["--out-root", str(self.out_root), "--config-dir", str(self.config_dir)]
        if self.panel_limit is not None:
            flags += ["--panel-limit", str(self.panel_limit)]
        if self.query_limit is not None:
            flags += ["--query-limit", str(self.query_limit)]
        if self.arms_filter:
            flags += ["--arms", ",".join(self.arms_filter)]
        if self.skip_upload:
            flags += ["--skip-upload"]
        if self.smoke:
            flags += ["--smoke"]
        flags += [
            "--tf-batch",
            str(self.tf_batch),
            "--null-cond",
            str(self.null_cond),
            "--null-prompt",
            str(self.null_prompt),
            "--perm-gate",
            str(self.perm_gate),
        ]
        return flags


# ── arm registry (the 52 content arms; metadata only) ─────────────────────────


def build_arm_rows() -> list[dict]:
    """The 52 content-arm rows in the #1979 arms.json row schema.

    theta_plus = the arm's FINAL trained state (root adapter == checkpoint-75;
    single-visit design). Marker arms are OUT of scope for this battery.
    mix_pos_path deliberately names a NONEXISTENT hub path: the anchor unit
    pre-writes the pos-FILTERED file locally (consumption-manifest ``pos:``
    rows only), so an accidental hub staging of the full mix fails loud
    instead of silently contaminating delta with negatives + generic rows.
    """
    import issue1947_cells as C47

    rows = []
    for cell in C47.CELLS:
        if cell.kind != "content":
            continue
        beh = cell.behavior
        src = f"icl_prefix_{beh}" if cell.ctx_key == "icl" else SRC_CONDITION_BY_CTX[cell.ctx_key]
        rows.append(
            {
                "arm_id": cell.slug,
                "kind": "content",
                "method": "lora",
                "beh_key": cell.beh_key,
                "behavior": beh,
                "ctx_key": cell.ctx_key,
                "regime": cell.regime,
                "visit": cell.visit,
                "seed": cell.seed,
                "adapter_repo": OVERFLOW_REPO,
                "adapter_subfolder": f"issue1947/{cell.slug}",
                "base_unit": "base_content",
                "mix_arm_id": cell.slug,
                "mix_layout": "i1947-pos-filtered",
                "mix_pos_path": f"{HF_PREFIX}/pos_filtered/{cell.slug}/pos_filtered.jsonl",
                "mix_hub_prefix": f"{MIX_BUCKET}/{cell.slug}",
                "source_condition": src,
            }
        )
    assert len(rows) == 52, len(rows)
    return rows


def build_config(cfg: Cfg) -> None:
    """VM-side config build: verbatim #1979 panel/queries + 52-arm arms.json +
    the 2 source-only extension members; sha-pinned manifest; uploaded to
    ``HF_PREFIX/config/`` so pod-side loads never depend on the git tree."""
    import shutil

    import issue1979_prep as PREP

    cfg.config_dir.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name in ("prefix_panel.json", "queries.json"):
        src = I1979_CONFIG / name
        assert src.is_file(), f"missing committed #1979 config: {src}"
        shutil.copyfile(src, cfg.config_dir / name)
        pins[name] = _sha256_file(src)
    _atomic_json(
        cfg.config_dir / "arms.json",
        {"arms": build_arm_rows(), "issue": ISSUE, "n_arms": 52, **_meta()},
    )
    ext = []
    for cid in EXT_SOURCE_CIDS:
        ctx = X.pfx_resolve_context(cid)
        m = PREP._member(
            cid,
            EXT_FAMILY,
            "trained artifact (source-only extension)",
            "registry render (pfx_resolve_context)",
            ctx,
            {"context_id": cid, "source_only": True},
        )
        ext.append(m)
    _atomic_json(cfg.config_dir / "ext_members.json", {"members": ext, **_meta()})
    pins["arms.json"] = _sha256_file(cfg.config_dir / "arms.json")
    pins["ext_members.json"] = _sha256_file(cfg.config_dir / "ext_members.json")
    _atomic_json(cfg.config_dir / "config_manifest.json", {"pins": pins, **_meta()})
    if not cfg.skip_upload:
        G._upload_paths(
            cfg.gcfg(),
            [cfg.config_dir / n for n in (*CONFIG_FILES_R3, "config_manifest.json")],
            f"{HF_PREFIX}/config",
        )
    logger.info("[build-config] wrote %s (pins: %s)", cfg.config_dir, list(pins))


def load_manifests(cfg: Cfg) -> dict:
    """Load + schema-assert the R3 config; slice at LOAD (smoke == full path).

    Extension members are appended AFTER the panel slice (they are source-only
    and must survive any --panel-limit so every icl arm keeps its C)."""
    for name in (*CONFIG_FILES_R3, "config_manifest.json"):
        p = cfg.config_dir / name
        if not p.exists():
            from explore_persona_space.orchestrate import hub

            logger.info("[config] staging %s from %s/config/", name, HF_PREFIX)
            hub.stage_hub_file(X.HF_DATA_REPO, f"{HF_PREFIX}/config/{name}", p, repo_type="dataset")
    pins = json.loads((cfg.config_dir / "config_manifest.json").read_text())["pins"]
    for name in CONFIG_FILES_R3:
        got = _sha256_file(cfg.config_dir / name)
        assert got == pins[name], f"config sha drift: {name} {got} != pinned {pins[name]}"
    panel = json.loads((cfg.config_dir / "prefix_panel.json").read_text())
    queries = json.loads((cfg.config_dir / "queries.json").read_text())
    arms = json.loads((cfg.config_dir / "arms.json").read_text())["arms"]
    ext = json.loads((cfg.config_dir / "ext_members.json").read_text())["members"]
    members = panel["members"]
    for m in members + ext:
        for k in ("prefix_id", "system", "prefix_turns", "user_wrap"):
            assert k in m, f"panel member missing key {k!r}: {sorted(m)}"
    qrows = queries["queries"]
    assert len(members) == 50 and len(qrows) == 60, (len(members), len(qrows))
    target_ids = [m["prefix_id"] for m in members]  # ORIGINAL 50 = the target universe
    if cfg.panel_limit is not None:
        members = members[: cfg.panel_limit]
        target_ids = target_ids[: cfg.panel_limit]
    if cfg.query_limit is not None:
        qrows = qrows[: cfg.query_limit]
    members = members + ext  # ext appended post-slice: source-only, never targets
    if cfg.arms_filter:
        keep = set(cfg.arms_filter)
        arms = [r for r in arms if r["arm_id"] in keep]
        assert len(arms) == len(keep), (sorted(keep - {r["arm_id"] for r in arms}), "unknown arm")
    member_ids = {m["prefix_id"] for m in members}
    for r in arms:
        assert r["source_condition"] in member_ids, (
            r["arm_id"],
            r["source_condition"],
            "source condition missing from sliced panel+ext — raise --panel-limit",
        )
    return {
        "members": members,
        "queries": qrows,
        "content_arms": arms,
        "marker_arms": [],
        "arm_rows": {r["arm_id"]: r for r in arms},
        "target_ids": target_ids,
        "panel_meta": {k: panel.get(k) for k in ("n_members", "seed", "limits", "pins")},
    }


# ── anchor prep: pos-filtered mix rows (delta inputs) ─────────────────────────


def prep_pos_filtered(cfg: Cfg, arm_id: str) -> Path:
    """Stage the arm's train_mix + consumption manifest, keep ONLY ``pos:`` rows
    (the behavior-bearing completions trained at C — the delta/t inputs), and
    write them where CAP._mix_positive_rows resolves its local file."""
    from explore_persona_space.orchestrate import hub

    dest = cfg.out_root / "delta_tf" / arm_id / "pos_filtered.jsonl"
    if dest.exists():
        return dest
    stage_dir = cfg.out_root / "mix_stage" / arm_id
    mix_p = stage_dir / "train_mix.jsonl"
    man_p = stage_dir / "consumption_manifest.json"
    meta_p = stage_dir / "mix_meta.json"
    for rel, local in (
        ("train_mix.jsonl", mix_p),
        ("consumption_manifest.json", man_p),
        ("mix_meta.json", meta_p),
    ):
        if not local.exists():
            hub.stage_hub_file(
                X.HF_DATA_REPO, f"{MIX_BUCKET}/{arm_id}/{rel}", local, repo_type="dataset"
            )
    rows = [json.loads(line) for line in mix_p.read_text().splitlines() if line.strip()]
    row_ids = json.loads(man_p.read_text())["row_ids"]
    meta = json.loads(meta_p.read_text())
    assert len(rows) == len(row_ids), (arm_id, len(rows), len(row_ids))
    pos = [r for r, rid in zip(rows, row_ids, strict=True) if str(rid).startswith("pos:")]
    assert len(pos) == meta["n_positive"], (arm_id, len(pos), meta["n_positive"])
    if cfg.smoke:
        pos = pos[:4]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pos))
    os.replace(tmp, dest)
    logger.info("[anchor-prep] %s: %d pos rows (of %d mix rows)", arm_id, len(pos), len(rows))
    return dest


# ── work items + dispatch (adapted from issue1979_gpu.dispatch) ───────────────


@dataclasses.dataclass
class Item:
    key: str
    phase: str
    deps: tuple[str, ...] = ()
    model_key: str | None = None
    heavy_model: bool = False


def build_work_items(cfg: Cfg, manifests: dict) -> list[Item]:
    arms = [r["arm_id"] for r in manifests["content_arms"]]
    items: list[Item] = [Item(key="gen:base_content", phase="gen")]
    for a in arms:
        items.append(Item(key=f"gen:{a}", phase="gen", model_key=a, heavy_model=True))
        items.append(
            Item(
                key=f"mt:{a}",
                phase="mt",
                deps=("gen:base_content",),
                model_key=a,
                heavy_model=True,
            )
        )
        items.append(Item(key=f"anchor:{a}", phase="anchor"))
    items.append(Item(key="battery2:sigma", phase="battery"))
    for a in arms:
        items.append(
            Item(
                key=f"battery2:{a}",
                phase="battery",
                deps=("gen:base_content", f"gen:{a}", f"mt:{a}", f"anchor:{a}", "battery2:sigma"),
            )
        )
    # summary sentinel keyed by arm-set size: a pilot-leg summary (1 arm) must
    # never satisfy the full run's aggregation (resume-regime key discipline)
    items.append(
        Item(
            key=f"battery2:summary:{len(arms)}",
            phase="battery",
            deps=tuple(f"battery2:{a}" for a in arms),
        )
    )
    items.append(Item(key="ladder:prep", phase="battery", deps=("gen:base_content",)))
    for a in arms:
        items.append(
            Item(
                key=f"ladder:{a}",
                phase="battery",
                deps=("gen:base_content", f"gen:{a}", f"mt:{a}", "ladder:prep"),
            )
        )
    items.append(
        Item(
            key=f"ladder:summary:{len(arms)}",
            phase="battery",
            deps=tuple(f"ladder:{a}" for a in arms),
        )
    )
    return items


def run_unit(cfg: Cfg, manifests: dict, key: str) -> list[str]:
    print(f"[phase={key.split(':')[0]} unit={key}]", flush=True)
    parts = key.split(":")
    gcfg = cfg.gcfg()
    if parts[0] == "gen":
        return G.run_f1a(gcfg, manifests, parts[1])
    if parts[0] == "mt":
        return G.run_f1b_writes(gcfg, manifests, parts[1], "base_content")
    if parts[0] == "anchor":
        prep_pos_filtered(cfg, parts[1])
        G.ensure_arm_registry(gcfg, manifests)
        return G.run_f1c(gcfg, manifests, parts[1])
    if parts[0] == "battery2" and parts[1] == "sigma":
        outs = run_battery_sigma(cfg)
        outs.append(str(build_psi_operator(cfg)))
        return outs
    if parts[0] == "battery2" and parts[1] == "summary":
        return run_battery_summary(cfg, manifests)
    if parts[0] == "battery2":
        return run_battery_arm(cfg, manifests, parts[1])
    if parts[0] == "ladder" and parts[1] == "prep":
        return run_ladder_prep(cfg, manifests)
    if parts[0] == "ladder" and parts[1] == "summary":
        return run_ladder_summary(cfg, manifests)
    if parts[0] == "ladder":
        return run_ladder_arm(cfg, manifests, parts[1])
    raise ValueError(f"unknown work-item key: {key}")


def dispatch(cfg: Cfg, manifests: dict, items: list[Item]) -> None:
    """Work-conserving round-robin over ALL visible GPUs (no wave barrier) —
    the issue1979_gpu.dispatch shape with this script as the worker target."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    pending = [it for it in items if not G._done(cfg, it.key)]
    logger.info("[dispatch] %d items, %d pending", len(items), len(pending))
    for ph in sorted({it.phase for it in pending}):
        n_ph = sum(1 for it in items if it.phase == ph)
        n_pend = sum(1 for it in pending if it.phase == ph)
        need = max(WORKER_HEADROOM_GB, PHASE_HEADROOM_GB[ph] * n_pend / max(n_ph, 1))
        assert_out_root_headroom(cfg.out_root, need, phase=ph)
    gpus = G._visible_gpus()
    if cfg.max_parallel is not None:
        gpus = gpus[: cfg.max_parallel]
    logger.info("[dispatch] %d workers (CVD pins: %s)", len(gpus), gpus)
    running: dict[str, tuple[subprocess.Popen, Item, float]] = {}
    done_keys = {it.key for it in items if G._done(cfg, it.key)}
    failures: list[str] = []
    exc_counts: dict[str, int] = {}
    abort_reason: str | None = None
    n_total, n_done = len(pending), 0
    script = str(Path(__file__).resolve())

    def _ready(it: Item) -> bool:
        if any(d not in done_keys for d in it.deps):
            return False
        heavy = sum(1 for _, r, _ in running.values() if r.heavy_model)
        if it.heavy_model and heavy >= MAX_HEAVY_CONCURRENT:
            return False
        if it.model_key and any(r.model_key == it.model_key for _, r, _ in running.values()):
            return False
        return True

    while pending or running:
        for gpu in gpus:
            if gpu in running or abort_reason:
                continue
            nxt = next((it for it in pending if _ready(it)), None)
            if nxt is None:
                continue
            pending.remove(nxt)
            cmd = [sys.executable, script, "--worker-unit", nxt.key, *cfg.worker_flags()]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}  # launcher-env CVD pin
            proc = subprocess.Popen(cmd, env=env)
            running[gpu] = (proc, nxt, time.time())
            logger.info("[dispatch] gpu%s <- %s (pid %d)", gpu, nxt.key, proc.pid)
        if not running:
            if pending and not failures and not abort_reason:
                raise RuntimeError(
                    f"deadlock: {len(pending)} pending, none ready: "
                    f"{[it.key for it in pending][:8]}"
                )
            break
        time.sleep(5.0)
        for gpu in list(running):
            proc, it, t0 = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            del running[gpu]
            wall = time.time() - t0
            if rc == 0 and G._done(cfg, it.key):
                done_keys.add(it.key)
                n_done += 1
                print(f"[r3] unit {n_done}/{n_total} {it.key} elapsed={wall:.0f}s", flush=True)
            else:
                exc = G._read_failure_class(cfg, it.key)
                failures.append(f"{it.key} rc={rc} exc={exc}")
                exc_counts[exc] = exc_counts.get(exc, 0) + 1
                if len(failures) > FAILURE_BUDGET:
                    abort_reason = f"failure budget exceeded ({len(failures)})"
                elif exc_counts[exc] >= SYSTEMIC_EXC_REPEAT:
                    abort_reason = f"systemic failure: {exc} x{exc_counts[exc]}"
                logger.error(
                    "[dispatch] FAILED %s rc=%s exc=%s after %.0fs (%d/%d budget)%s",
                    it.key,
                    rc,
                    exc,
                    wall,
                    len(failures),
                    FAILURE_BUDGET,
                    f" — ABORT: {abort_reason}" if abort_reason else "",
                )
    if failures:
        skipped = [it.key for it in pending]
        raise RuntimeError(
            f"R3 units failed ({len(failures)} of {n_total}"
            + (f"; ABORTED: {abort_reason}" if abort_reason else "")
            + (f"; {len(skipped)} never scheduled: {skipped[:8]}" if skipped else "")
            + f"): {failures}"
        )
    terminal = "smoke_done" if cfg.smoke else cfg.terminal_token
    print(f"[phase={terminal}]", flush=True)  # noqa: phase-done-reserved
    _atomic_json(
        cfg.out_root / "r3_results.json",
        {"issue": ISSUE, "phase": "r3", "status": terminal, "n_items": len(items), **_meta()},
    )


# ── battery: shared loaders ───────────────────────────────────────────────────


def _ensure_local(cfg: Cfg, rel: str) -> Path:
    """out_root-local artifact, staged from HF_PREFIX on miss (VM re-runs)."""
    from explore_persona_space.orchestrate import hub

    p = cfg.out_root / rel
    if not p.exists():
        hub.stage_hub_file(X.HF_DATA_REPO, f"{HF_PREFIX}/{rel}", p, repo_type="dataset")
    return p


def _load_store(cfg: Cfg, rel: str) -> dict:
    import torch

    return torch.load(_ensure_local(cfg, rel), map_location="cpu", weights_only=False)


def _corpus_train_rows(cfg: Cfg, layer: int, span: str) -> "object":
    """Corpus TRAIN rows for one span at one layer (the corpus_sigma slicing)."""
    import numpy as np
    import torch

    store = torch.load(
        cfg.out_root / "corpus_capture" / "base_content" / "pooled.pt",
        map_location="cpu",
        weights_only=False,
    )
    mat = np.asarray(store["arms"][span][layer].float().numpy(), dtype=np.float64)
    sample = X.load_corpus_sample(cfg.out_root)
    qidx = np.asarray(store["row_question_idx"])
    return mat[qidx < sample["n_train"]]


def build_psi_operator(cfg: Cfg) -> Path:
    """psi := ridge-regularized pseudo-inverse of the banked base map M0 (L19).

    The paper leaves psi undefined; PINNED (stated deviation) as
    psi_lambda(y) = unstd(SVD-ridge-inverse of M0's W applied to (y - ymu)).
    lambda is selected on corpus TRAIN rows only (never gate-race targets), by
    minimizing || z_tr - psi_z(v0_tr) ||_F^2 against the ACTUAL base response
    rows (criterion: invert real (c0, v0) pairs, not the map's own outputs).
    Reports selected lambda + effective rank sum(S^2/(S^2+lambda))."""
    import numpy as np

    from explore_persona_space.orchestrate import hub

    out = cfg.out_root / "battery2" / f"psi_L{PSI_LAYER}.npz"
    if out.exists():
        return out
    m0_local = cfg.out_root / "battery2" / "m0_L19.pt"
    if not m0_local.exists():
        hub.stage_hub_file(
            X.HF_DATA_REPO, M0_HUB_PATH, m0_local, repo_type="dataset", revision=G.I1900_PIN
        )
    import torch

    payload = torch.load(m0_local, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
        payload = payload["payload"]
    for k in ("W", "xmu", "xsd", "ymu"):
        assert k in payload, (sorted(payload), "m0 payload schema drift")
    W = np.asarray(payload["W"], dtype=np.float64)
    xmu = np.asarray(payload["xmu"], dtype=np.float64).ravel()
    xsd = np.asarray(payload["xsd"], dtype=np.float64).ravel()
    ymu = np.asarray(payload["ymu"], dtype=np.float64).ravel()
    U, S, Vt = np.linalg.svd(W, full_matrices=False)  # z @ W = z @ U S Vt
    C_tr = _corpus_train_rows(cfg, PSI_LAYER, "context")
    V0_tr = _corpus_train_rows(cfg, PSI_LAYER, "response")
    z_tr = (C_tr - xmu) / xsd
    y_tr = V0_tr - ymu
    yV = y_tr @ Vt.T  # (n, r)
    zU = z_tr @ U  # (n, r) — target in the U basis (exact: ||z - zhat|| decomposes)
    errs = {}
    for lam in PSI_LAMBDAS:
        shrink = S / (S**2 + lam)
        zhat_U = yV * shrink
        # || z - zhat ||^2 = || zU - zhat_U ||^2 + const (U orthonormal, full rank)
        errs[lam] = float(((zU - zhat_U) ** 2).sum())
    lam_star = min(errs, key=errs.get)
    eff_rank = float((S**2 / (S**2 + lam_star)).sum())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            U=U,
            S=S,
            Vt=Vt,
            xmu=xmu,
            xsd=xsd,
            ymu=ymu,
            lam_star=lam_star,
            eff_rank=eff_rank,
            n_train_rows=z_tr.shape[0],
            errs_grid=np.array([[lam, errs[lam]] for lam in PSI_LAMBDAS]),
        )
    os.replace(tmp, out)
    logger.info(
        "[psi] L%d lambda*=%.4g eff_rank=%.1f (criterion: invert actual (c0,v0) train rows)",
        PSI_LAYER,
        lam_star,
        eff_rank,
    )
    return out


def _psi_apply(psi, y: "object") -> "object":
    """psi(y): answer-space vector -> context-space vector (raw, unstandardized)."""
    shrink = psi["S"] / (psi["S"] ** 2 + float(psi["lam_star"]))
    z = ((y - psi["ymu"]) @ psi["Vt"].T) * shrink @ psi["U"].T
    return z * psi["xsd"] + psi["xmu"]


def _m0_predict(cfg: Cfg, c_vec: "object") -> "object":
    """Banked-M0 forward prediction for one raw context vector (L19)."""
    import torch

    payload = torch.load(
        cfg.out_root / "battery2" / "m0_L19.pt", map_location="cpu", weights_only=False
    )
    pred = G._apply_saved_map(payload, c_vec[None, :], "cpu")
    import numpy as np

    return np.asarray(pred, dtype=np.float64)[0]


def run_battery_sigma(cfg: Cfg) -> list[str]:
    """Corpus second moment Sigma_c per layer (the #1768 recipe: 15k bare TRAIN
    context span-means at the pinned corpus revision; shrinkage 0.1)."""
    import numpy as np

    import issue1768_directions as DIR
    from explore_persona_space.orchestrate import hub

    outs = []
    stages = [
        (f"{X.HF_PREFIX}/inputs/corpus_sample.json", "inputs/corpus_sample.json", G.CORPUS_PIN),
        (
            f"{X.HF_PREFIX}/corpus_capture/base_content/pooled.pt",
            "corpus_capture/base_content/pooled.pt",
            G.CORPUS_PIN,
        ),
    ]
    for rel_hub, rel_local, rev in stages:
        p = cfg.out_root / rel_local
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            hub.stage_hub_file(X.HF_DATA_REPO, rel_hub, p, repo_type="dataset", revision=rev)
    for li in LAYERS:
        out = cfg.out_root / "battery2" / f"sigma_L{li}.npz"
        if out.exists():
            outs.append(str(out))
            continue
        sig = DIR.corpus_sigma(cfg.out_root, li)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, sigma=sig["sigma"], chol=sig["chol"], n_rows=sig["n_rows"])
        os.replace(tmp, out)
        outs.append(str(out))
        logger.info("[battery:sigma] L%d from %d corpus train rows", li, sig["n_rows"])
    return outs


def _rows_index(store: dict) -> dict[str, int]:
    return {sha: i for i, sha in enumerate(store["row_sha"])}


def _profiles(
    store: dict,
    layer: int,
    span: str,
    keep_shas: set[str] | None = None,
    pos: str | None = None,
) -> dict[str, "object"]:
    """prefix_id -> (mean vec float64, query_sha set) over kept rows.

    ``span`` reads store['spans'] (span-mean pooling); ``pos`` reads
    store['positions'] (token-position pooling) instead."""
    import numpy as np

    src = store["positions"][pos][layer] if pos else store["spans"][span][layer]
    mat = np.asarray(src.float().numpy(), dtype=np.float64)
    by_prefix: dict[str, list[int]] = {}
    qsets: dict[str, set[str]] = {}
    for i, (pid, qsha) in enumerate(
        zip(store["row_prefix_id"], store["row_query_sha"], strict=True)
    ):
        if keep_shas is not None and store["row_sha"][i] not in keep_shas:
            continue
        by_prefix.setdefault(pid, []).append(i)
        qsets.setdefault(pid, set()).add(qsha)
    return {pid: (mat[idx].mean(axis=0), qsets[pid]) for pid, idx in by_prefix.items()}


def _paired_condition_deltas(
    arm_store: dict, base_store: dict, layer: int, span: str, query_shas: list[str]
) -> tuple[dict[str, "object"], dict]:
    """On-policy condition deltas: per condition, mean_arm - mean_base over the
    INTERSECTION of (query_sha in subset) rows present in both stores; floors
    asserted per condition (>= QUERY_FLOOR_FRAC of the subset)."""
    import numpy as np

    qsel = set(query_shas)
    floor = max(1, int(np.ceil(QUERY_FLOOR_FRAC * len(query_shas))))
    arm_rows: dict[tuple[str, str], int] = {}
    for i, (pid, qsha) in enumerate(
        zip(arm_store["row_prefix_id"], arm_store["row_query_sha"], strict=True)
    ):
        if qsha in qsel:
            arm_rows[(pid, qsha)] = i
    base_rows: dict[tuple[str, str], int] = {}
    for i, (pid, qsha) in enumerate(
        zip(base_store["row_prefix_id"], base_store["row_query_sha"], strict=True)
    ):
        if qsha in qsel:
            base_rows[(pid, qsha)] = i
    a_mat = np.asarray(arm_store["spans"][span][layer].float().numpy(), dtype=np.float64)
    b_mat = np.asarray(base_store["spans"][span][layer].float().numpy(), dtype=np.float64)
    deltas: dict[str, np.ndarray] = {}
    realized: dict[str, int] = {}
    pids = {pid for pid, _ in base_rows}
    for pid in pids:
        common = sorted(q for (p, q) in arm_rows if p == pid and (pid, q) in base_rows)
        assert len(common) >= floor, (
            pid,
            len(common),
            floor,
            "per-condition realized-query floor violated (dropped rows?)",
        )
        ai = [arm_rows[(pid, q)] for q in common]
        bi = [base_rows[(pid, q)] for q in common]
        deltas[pid] = a_mat[ai].mean(axis=0) - b_mat[bi].mean(axis=0)
        realized[pid] = len(common)
    return deltas, {"realized_queries": realized, "floor": floor}


def _prompt_deltas(arm_store: dict, base_store: dict, layer: int, span: str) -> "object":
    """Per-prompt deltas paired by row_sha: (n_rows, d) float64."""
    import numpy as np

    ai = _rows_index(arm_store)
    bi = _rows_index(base_store)
    common = [s for s in base_store["row_sha"] if s in ai]
    a_mat = np.asarray(arm_store["spans"][span][layer].float().numpy(), dtype=np.float64)
    b_mat = np.asarray(base_store["spans"][span][layer].float().numpy(), dtype=np.float64)
    return a_mat[[ai[s] for s in common]] - b_mat[[bi[s] for s in common]]


# ── battery: statistics ───────────────────────────────────────────────────────


def _rank_stats(delta_V: "object") -> dict:
    """Spectrum stats of the (d x n) stacked update. UNCENTERED is the paper's
    own statistic (the rank-one structure INCLUDES the mean); centered reported
    as the mean-removed remainder (block D's split)."""
    import numpy as np

    d, n = delta_V.shape
    sv = np.linalg.svd(delta_V, compute_uv=False)
    s2 = sv**2
    U, _, _ = np.linalg.svd(delta_V, full_matrices=False)
    mean_col = delta_V.mean(axis=1)
    centered = delta_V - mean_col[:, None]
    sv_c = np.linalg.svd(centered, compute_uv=False)
    s2_c = sv_c**2
    return {
        "n_cols": int(n),
        "top1_var_share": float(s2[0] / s2.sum()),
        "sigma2_over_sigma1": float(sv[1] / sv[0]) if len(sv) > 1 else float("nan"),
        "top1_var_share_centered": float(s2_c[0] / s2_c.sum()),
        "sv_top10": [float(v) for v in sv[:10]],
        "u1": U[:, 0],
        "mean_col_norm": float(np.linalg.norm(mean_col)),
        "centered_fro": float(np.linalg.norm(centered)),
        "fro": float(np.linalg.norm(delta_V)),
    }


def _null_top1_condition(col_norms: "object", d: int, draws: int, rng) -> dict:
    """n-matched null: iid gaussian columns rescaled to the realized column
    norms; exact top-1 share via the (n x n) Gram eigenspectrum, batched."""
    import numpy as np

    n = len(col_norms)
    top1 = np.empty(draws)
    batch = max(1, min(draws, int(2e8 / (d * n * 8))))
    done = 0
    while done < draws:
        b = min(batch, draws - done)
        Z = rng.standard_normal((b, d, n))
        Z *= (np.asarray(col_norms) / np.linalg.norm(Z, axis=1))[:, None, :]
        gram = np.einsum("bdi,bdj->bij", Z, Z)
        ev = np.linalg.eigvalsh(gram)
        top1[done : done + b] = ev[:, -1] / ev.sum(axis=1)
        done += b
    return {
        "mean": float(top1.mean()),
        "p2_5": float(np.quantile(top1, 0.025)),
        "p97_5": float(np.quantile(top1, 0.975)),
        "n_draws": int(draws),
    }


def _null_top1_prompt(col_norms: "object", d: int, draws: int, rng) -> dict:
    """n-matched null at prompt grain (n ~ 3000): top sigma via low-rank SVD on
    GPU when available; Fro^2 = sum of column norms^2 exactly."""
    import numpy as np
    import torch

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norms = torch.as_tensor(np.asarray(col_norms), dtype=torch.float32, device=dev)
    fro2 = float((norms**2).sum().item())
    top1 = []
    for k in range(draws):
        g = torch.Generator(device="cpu").manual_seed(SEED + 7919 * k)
        Z = torch.randn(d, len(col_norms), generator=g, dtype=torch.float32).to(dev)
        Z *= norms / Z.norm(dim=0)
        _u, s, _v = torch.svd_lowrank(Z, q=8, niter=4)
        top1.append(float(s[0].item() ** 2) / fro2)
    arr = np.asarray(top1)
    return {
        "mean": float(arr.mean()),
        "p2_5": float(np.quantile(arr, 0.025)),
        "p97_5": float(np.quantile(arr, 0.975)),
        "n_draws": int(draws),
    }


def _cv_rank_curve(dva: dict, dvb: dict, target_ids: list[str]) -> dict:
    """Query-aligned split-half CV of the update's DIRECTIONS: SVD basis from
    half-A condition deltas, reconstruction error on half-B deltas, m=0..CV_M_MAX.
    Noise floor = ||B - A||_F / ||B||_F (half-vs-half disagreement)."""
    import numpy as np

    ids = [t for t in target_ids if t in dva and t in dvb]
    A = np.stack([dva[t] for t in ids], axis=1)
    B = np.stack([dvb[t] for t in ids], axis=1)
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    b_norm = np.linalg.norm(B)
    errs = [1.0]
    m_max = min(CV_M_MAX, U.shape[1])
    for m in range(1, m_max + 1):
        Um = U[:, :m]
        resid = B - Um @ (Um.T @ B)
        errs.append(float(np.linalg.norm(resid) / b_norm))
    m_star = m_max
    for m in range(1, m_max + 1):
        if m == m_max or errs[m] - errs[m + 1] < CV_ELBOW_EPS:
            m_star = m
            break
    noise_floor = float(np.linalg.norm(B - A) / b_norm)
    within = [m for m in range(1, m_max + 1) if errs[m] <= noise_floor * 1.10]
    return {
        "cv_error_by_m": errs,  # index = m (errs[0] == 1.0 baseline)
        "m_star_elbow": int(m_star),
        "m_star_noise10": int(within[0]) if within else None,
        "noise_floor": noise_floor,
        "n_cols": len(ids),
    }


def _spearman_perm(g_pred: "object", g_hat: "object", draws: int, rng) -> dict:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    g_pred = np.asarray(g_pred, dtype=np.float64)
    g_hat = np.asarray(g_hat, dtype=np.float64)
    rho = float(spearmanr(g_pred, g_hat).statistic)
    r = float(pearsonr(g_pred, g_hat).statistic)
    sign_agree = float(np.mean(np.sign(g_pred) == np.sign(g_hat)))
    mae = float(np.mean(np.abs(g_pred - g_hat)))

    # permutation null on the spearman: permute g_hat across conditions
    def _ranks(v):
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(v))
        return ranks

    rp, rh = _ranks(g_pred), _ranks(g_hat)
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)
    rh = (rh - rh.mean()) / (rh.std() + 1e-12)
    n = len(rh)
    null = np.empty(draws)
    for k in range(draws):
        null[k] = float(rp @ rh[rng.permutation(n)]) / n
    p_emp = float((np.abs(null) >= abs(rho)).mean())
    return {
        "spearman": rho,
        "pearson": r,
        "sign_agreement": sign_agree,
        "mae": mae,
        "perm_p": p_emp,
        "perm_abs_p95": float(np.quantile(np.abs(null), 0.95)),
        "n": int(n),
    }


def run_battery_arm(cfg: Cfg, manifests: dict, arm_id: str) -> list[str]:
    """Blocks A-D for one arm, both trees, layers 14/19/25."""
    import numpy as np

    import issue1768_directions as DIR

    row = manifests["arm_rows"][arm_id]
    src_id = row["source_condition"]
    # duplicate-condition drop: neg_default_assistant renders byte-identical to
    # `bare` (verified content-sha match) — captured but never a target.
    target_ids = [t for t in manifests["target_ids"] if t != src_id and t not in DUP_TARGET_DROP]
    queries = manifests["queries"]
    q_shas = [q["sha"] for q in queries]
    assert len(q_shas) >= 2, "need >= 2 queries for the aligned split-half CV"
    half_a = [s for i, s in enumerate(q_shas) if i % 2 == 0]
    half_b = [s for i, s in enumerate(q_shas) if i % 2 == 1]
    base = _load_store(cfg, "stores/onpolicy/base_content/store.pt")
    own = _load_store(cfg, f"stores/onpolicy/{arm_id}/store.pt")
    mt = _load_store(cfg, f"stores/matched_tf/{arm_id}/store.pt")
    anch_p = cfg.out_root / "anchors" / arm_id / "anchors.pt"
    if not anch_p.exists():
        _ensure_local(cfg, f"anchors/{arm_id}/anchors.pt")
    import torch

    anchors = torch.load(anch_p, map_location="cpu", weights_only=False)
    rb = DIR.load_rb_tensors(cfg.out_root)
    psi = np.load(cfg.out_root / "battery2" / f"psi_L{PSI_LAYER}.npz")
    rng = np.random.default_rng(SEED)
    out: dict = {
        "arm": row,
        "trees": {},
        "dup_targets_dropped": list(DUP_TARGET_DROP),
        "psi": {
            "definition": "ridge-pinv of banked M0 (L19); paper leaves psi undefined",
            "lambda_star": float(psi["lam_star"]),
            "effective_rank": float(psi["eff_rank"]),
        },
        **_meta(),
    }
    for tree, arm_store in (("onpolicy", own), ("matched_text", mt)):
        per_layer: dict = {}
        for li in LAYERS:
            dv_full, cov = _paired_condition_deltas(arm_store, base, li, "response", q_shas)
            dv_a, _ = _paired_condition_deltas(arm_store, base, li, "response", half_a)
            dv_b, _ = _paired_condition_deltas(arm_store, base, li, "response", half_b)
            assert src_id in dv_full, (arm_id, src_id, "source condition has no delta")
            w_hat = dv_full[src_id]
            tids = [t for t in target_ids if t in dv_full]
            dV = np.stack([dv_full[t] for t in tids], axis=1)  # (d, n_targets)
            d = dV.shape[0]
            # A: realized gates + scalarity residuals + spectrum + CV curve
            ww = float(w_hat @ w_hat)
            g_hat = dV.T @ w_hat / (ww + 1e-12)
            resid = dV - np.outer(w_hat, g_hat)
            scalarity = np.linalg.norm(resid, axis=0) / (np.linalg.norm(dV, axis=0) + 1e-12)
            rank = _rank_stats(dV)
            u1 = rank.pop("u1")
            rank["cos_u1_w"] = float(
                abs(u1 @ w_hat) / (np.linalg.norm(u1) * np.linalg.norm(w_hat) + 1e-12)
            )
            rank["null"] = _null_top1_condition(np.linalg.norm(dV, axis=0), d, cfg.null_cond, rng)
            cv = _cv_rank_curve(dv_a, dv_b, tids)
            # prompt grain
            pd = _prompt_deltas(arm_store, base, li, "response")
            prank = _rank_stats(pd.T)
            prank.pop("u1")
            if li == PRIMARY_LAYER:
                prank["null"] = _null_top1_prompt(
                    np.linalg.norm(pd, axis=1), d, cfg.null_prompt, rng
                )
            # B: source-write direction (delta from anchors; base profile at C)
            a_l = anchors[f"L{li}"]
            t_vec = np.asarray(a_l["A_ans"].float().numpy(), dtype=np.float64)
            base_prof = _profiles(base, li, "response")
            v0_src = base_prof[src_id][0]
            delta = t_vec - v0_src
            dd = float(delta @ delta)
            eta = float(delta @ w_hat) / (dd + 1e-12)
            rel_res = float(np.linalg.norm(w_hat - eta * delta) / (np.linalg.norm(w_hat) + 1e-12))
            sig_npz = np.load(cfg.out_root / "battery2" / f"sigma_L{li}.npz")
            sigma = {"sigma": sig_npz["sigma"], "chol": sig_npz["chol"]}
            DIR.N_NULL_DRAWS = cfg.perm_gate  # null band draws (smoke-scaled)
            bands = DIR.null_bands(w_hat, sigma, rng)
            beh_rb = rb[row["beh_key"]]
            assert beh_rb.shape[0] > max(LAYERS), (row["beh_key"], beh_rb.shape)
            rbl = beh_rb[li]

            def _cos(a, b):
                return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

            # three-object family (kept DISTINCT — the theory's delta is NOT
            # #1768's map residual): t_bar (banked base TF over mix positives),
            # delta_theory = t_bar - v0(C) (base PROFILE at the source condition,
            # this panel, condition grain), map_residual = t_bar - M0(c_C)
            # (what #1768's operator-KV read actually raced; L19 only — the
            # banked M0's layer). One norm-matched null band covers all three
            # cosines (the band is over random directions vs w_hat).
            dirs = {
                "cos_w_delta": _cos(w_hat, delta),
                "cos_w_tbar": _cos(w_hat, t_vec),
                "eta_hat": eta,
                "rel_residual": rel_res,
                "cos_w_rb": _cos(w_hat, rbl),
                "delta_norm": float(np.sqrt(dd)),
                "tbar_norm": float(np.linalg.norm(t_vec)),
                "w_norm": float(np.sqrt(ww)),
                "null_bands": bands,
                "anchor_n_rows": int(anchors["n_rows"]),
            }
            map_resid = None
            if li == PSI_LAYER:
                ctx_prof_l19 = _profiles(base, li, "context")
                map_resid = t_vec - _m0_predict(cfg, ctx_prof_l19[src_id][0])
                dirs["cos_w_map_residual"] = _cos(w_hat, map_resid)
                dirs["map_residual_norm"] = float(np.linalg.norm(map_resid))
            # C: the gate race, two q poolings. psi HEADLINE = ridge-pinv of the
            # banked M0 (L19 only); psi = identity kept as a LABELED sensitivity
            # arm at every layer (type confusion: answer-space vector used as a
            # context-space key — dimensions coincide, semantics do not).
            ctx_prof = _profiles(base, li, "context")
            lp_prof = _profiles(base, li, "context", pos="last_prompt")
            gates: dict = {}
            for pooling, prof in (("span_mean", ctx_prof), ("last_prompt", lp_prof)):
                c_src = prof[src_id][0]
                Q = np.stack([prof[t][0] for t in tids], axis=1)
                keys = {
                    "c_C": c_src,
                    "psiId_t": t_vec,
                    "c_C_plus_psiId_delta": c_src + delta,
                    "psiId_delta": delta,  # paper's 4th ablation key (supplement)
                }
                if li == PSI_LAYER:
                    psi_t = _psi_apply(psi, t_vec)
                    psi_d = _psi_apply(psi, delta)
                    keys |= {
                        "psiPinv_t": psi_t,
                        "c_C_plus_psiPinv_delta": c_src + psi_d,
                        "psiPinv_delta": psi_d,  # supplement
                    }
                metrics = {"I": None}
                if pooling == "span_mean":
                    metrics["whitened"] = sigma  # Sigma bank is span-mean pooled only
                for kname, kvec in keys.items():
                    for mname, mm in metrics.items():
                        if mm is None:
                            kM = kvec
                        else:  # reuse the stored Cholesky factor (one factorization per layer)
                            from scipy.linalg import cho_solve

                            kM = cho_solve((mm["chol"], True), kvec)
                        denom = float(kM @ c_src)
                        g_pred = (kM @ Q) / (denom + 1e-12)
                        gates[f"{pooling}|{kname}|{mname}"] = _spearman_perm(
                            g_pred, g_hat, cfg.perm_gate, rng
                        ) | {"denom_kMqC": denom}
            per_layer[f"L{li}"] = {
                "coverage": cov,
                "w_norm": float(np.sqrt(ww)),
                "gate_realized": {t: float(g) for t, g in zip(tids, g_hat, strict=True)},
                "scalarity_residual": {t: float(s) for t, s in zip(tids, scalarity, strict=True)},
                "rank_condition": rank,
                "cv": cv,
                "rank_prompt": prank,
                "direction": dirs,
                "gates": gates,
                "operator_block": {
                    "map_fit": "SKIPPED",
                    "n_train_condition": len(tids),
                    "n_train_prompt": int(pd.shape[0]),
                    "d": int(d),
                    "reason": "n_train < d at both grains (under-determined)",
                    "mean_shift_norm": rank["mean_col_norm"],
                    "centered_fro": rank["centered_fro"],
                },
            }
        out["trees"][tree] = per_layer
    dest = cfg.out_root / "battery2" / "arms" / f"{arm_id}.json"
    _atomic_json(dest, out)
    G._upload_paths(cfg.gcfg(), [dest], f"{HF_PREFIX}/battery2/arms")
    return [str(dest)]


def run_battery_summary(cfg: Cfg, manifests: dict) -> list[str]:
    """Aggregate the per-arm battery JSONs into summary.json (headline: the
    cross-validated m per arm/behavior/regime, on-policy primary)."""
    import numpy as np

    arms = [r["arm_id"] for r in manifests["content_arms"]]
    per_arm = {}
    for a in arms:
        p = cfg.out_root / "battery2" / "arms" / f"{a}.json"
        if not p.exists():
            _ensure_local(cfg, f"battery/arms/{a}.json")
        per_arm[a] = json.loads(p.read_text())
    key = f"L{PRIMARY_LAYER}"
    summary: dict = {"n_arms": len(arms), "primary_layer": PRIMARY_LAYER, **_meta()}
    for tree in ("onpolicy", "matched_text"):
        rows = []
        for a, rec in per_arm.items():
            L = rec["trees"][tree][key]
            arm_meta = rec["arm"]
            rows.append(
                {
                    "arm": a,
                    "behavior": arm_meta["behavior"],
                    "ctx": arm_meta["ctx_key"],
                    "regime": arm_meta["regime"],
                    "visit": arm_meta["visit"],
                    "seed": arm_meta["seed"],
                    "m_star_elbow": L["cv"]["m_star_elbow"],
                    "m_star_noise10": L["cv"]["m_star_noise10"],
                    "noise_floor": L["cv"]["noise_floor"],
                    "cv_error_by_m": L["cv"]["cv_error_by_m"],
                    "top1_uncentered": L["rank_condition"]["top1_var_share"],
                    "top1_centered": L["rank_condition"]["top1_var_share_centered"],
                    "top1_null_p97_5": L["rank_condition"]["null"]["p97_5"],
                    "sigma2_over_sigma1": L["rank_condition"]["sigma2_over_sigma1"],
                    "cos_u1_w": L["rank_condition"]["cos_u1_w"],
                    "top1_prompt": L["rank_prompt"]["top1_var_share"],
                    "top1_prompt_null_p97_5": (L["rank_prompt"].get("null") or {}).get("p97_5"),
                    "scalarity_median": float(np.median(list(L["scalarity_residual"].values()))),
                    "scalarity_p90": float(
                        np.quantile(list(L["scalarity_residual"].values()), 0.9)
                    ),
                    "cos_w_delta": L["direction"]["cos_w_delta"],
                    "cos_w_tbar": L["direction"]["cos_w_tbar"],
                    "cos_w_map_residual": L["direction"].get("cos_w_map_residual"),
                    "eta_hat": L["direction"]["eta_hat"],
                    "rel_residual": L["direction"]["rel_residual"],
                    "cos_w_rb": L["direction"]["cos_w_rb"],
                    "iso_null_abs_p95": L["direction"]["null_bands"]["isotropic"]["abs_p95"],
                    "cov_null_abs_p95": L["direction"]["null_bands"]["corpus_covariance"][
                        "abs_p95"
                    ],
                    "gates": {
                        k: {kk: v[kk] for kk in ("spearman", "pearson", "perm_p")}
                        for k, v in L["gates"].items()
                    },
                }
            )
        summary[tree] = rows
    summary["panel_note"] = (
        "targets = the verbatim #1979 50-member panel MINUS neg_default_assistant "
        "(render-identical duplicate of `bare`) MINUS each arm's own source condition; "
        "two SOURCE-ONLY extension members (icl_prefix_impolite, icl_prefix_writing_style) "
        "serve the 8 imp/cas ICL arms' w_hat only and never enter any target set "
        "(stated deviation from the capture-all-52 redirect; per-arm realized-n nulls "
        "mediate every cross-arm rank comparison). The 44 non-ICL arms never see the two "
        "extension prefixes as targets — the cross-behavior swapped-prefix probe is a "
        "named coverage gap."
    )
    dest = cfg.out_root / "battery2" / "summary.json"
    _atomic_json(dest, summary)
    G._upload_paths(cfg.gcfg(), [dest], f"{HF_PREFIX}/battery2")
    return [str(dest)]


# ── ablation ladder (addendum): context-vs-map at condition grain ─────────────


def _joined_rows(arm_store: dict, base_store: dict, layer: int, q_shas: list[str]) -> dict:
    """Row-aligned (c0, v0, cp, vp) arrays over (prefix, query) keys present in
    BOTH stores, restricted to the query subset. Returns float64 (n, d) arrays."""
    import numpy as np

    qsel = set(q_shas)

    def _index(store):
        return {
            (pid, qsha): i
            for i, (pid, qsha) in enumerate(
                zip(store["row_prefix_id"], store["row_query_sha"], strict=True)
            )
            if qsha in qsel
        }

    bi, ai = _index(base_store), _index(arm_store)
    keys = sorted(k for k in bi if k in ai)
    assert keys, "ladder join produced zero rows"
    b_ctx = np.asarray(base_store["spans"]["context"][layer].float().numpy(), dtype=np.float64)
    b_res = np.asarray(base_store["spans"]["response"][layer].float().numpy(), dtype=np.float64)
    a_ctx = np.asarray(arm_store["spans"]["context"][layer].float().numpy(), dtype=np.float64)
    a_res = np.asarray(arm_store["spans"]["response"][layer].float().numpy(), dtype=np.float64)
    bidx = [bi[k] for k in keys]
    aidx = [ai[k] for k in keys]
    return {
        "keys": keys,
        "c0": b_ctx[bidx],
        "v0": b_res[bidx],
        "cp": a_ctx[aidx],
        "vp": a_res[aidx],
    }


def _pca_fit(X_train: "object", k: int) -> dict:
    """Train-fold PCA of context rows: mean + top-k components + retained var."""
    import numpy as np

    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    k_real = int(min(k, vt.shape[0]))
    var = s**2
    return {
        "mu": mu,
        "components": vt[:k_real].T,  # (d, k_real)
        "k_requested": int(k),
        "k_real": k_real,
        "retained_var_frac": float(var[:k_real].sum() / (var.sum() + 1e-12)),
    }


def _pca_apply(pca: dict, X: "object") -> "object":
    return (X - pca["mu"]) @ pca["components"]


def run_ladder_prep(cfg: Cfg, manifests: dict) -> list[str]:
    """Shared base-side ladder prep per (layer, k): train-fold PCA of BASE
    context rows + the shared M0 ridge fit (PCA(c0_train) -> v0_train). M0 is
    fit ONCE and shared across arms/trees (the addendum's estimator rule)."""
    import torch

    import issue825_map_alignment as MA

    q_shas = [q["sha"] for q in manifests["queries"]]
    train_q = [s for i, s in enumerate(q_shas) if i % 2 == 0]  # the aligned partition
    base = _load_store(cfg, "stores/onpolicy/base_content/store.pt")
    outs = []
    for li in LAYERS:
        j = _joined_rows(base, base, li, train_q)  # base joined with itself: c0/v0 train rows
        for k in LADDER_KS:
            out = cfg.out_root / "ladder" / f"prep_L{li}_k{k}.pt"
            if out.exists():
                outs.append(str(out))
                continue
            pca = _pca_fit(j["c0"], k)
            X_tr = torch.as_tensor(_pca_apply(pca, j["c0"]), dtype=torch.float64)
            Y_tr = torch.as_tensor(j["v0"], dtype=torch.float64)
            prep = MA._ridge_prep(X_tr)
            # materialize the selected lambda once via the memoized predict path
            _ = MA._ridge_predict(prep, Y_tr, X_tr[:1], lam_key="m0")
            lam_m0 = float(prep["_lam_memo"]["m0"][2])
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".pt.tmp")
            torch.save(
                {
                    "pca": pca,
                    "prep": prep,
                    "Y_train_v0": Y_tr,
                    "lambda_m0": lam_m0,
                    "n_train": int(X_tr.shape[0]),
                    "layer": li,
                    "k": k,
                    "metadata": _meta(),
                },
                tmp,
            )
            os.replace(tmp, out)
            logger.info(
                "[ladder:prep] L%d k%d n_train=%d retained_var=%.3f lambda_m0=%.4g",
                li,
                k,
                int(X_tr.shape[0]),
                pca["retained_var_frac"],
                lam_m0,
            )
            outs.append(str(out))
    return outs


def _pooled_r2(pred: "object", truth: "object", baseline_mean: "object") -> float:

    num = float(((pred - truth) ** 2).sum())
    den = float(((truth - baseline_mean) ** 2).sum())
    return 1.0 - num / (den + 1e-12)


def run_ladder_arm(cfg: Cfg, manifests: dict, arm_id: str) -> list[str]:
    """Four-rung context-vs-map ablation for one arm: M0c0 / M0c+ / M+c0 / M+c+,
    held-out pooled R^2 vs the measured v+, per tree x layer x k. TREE
    COHERENCE: each tree's M+ fits that tree's own (c+, v+) rows; M0 is the
    shared base fit. kNN retrieval attached per fitted map; the identity+bias
    baseline is stated INAPPLICABLE (PCA-k inputs vs full-d outputs)."""
    import numpy as np
    import torch

    import issue825_map_alignment as MA
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    q_shas = [q["sha"] for q in manifests["queries"]]
    train_q = [s for i, s in enumerate(q_shas) if i % 2 == 0]
    eval_q = [s for i, s in enumerate(q_shas) if i % 2 == 1]
    base = _load_store(cfg, "stores/onpolicy/base_content/store.pt")
    own = _load_store(cfg, f"stores/onpolicy/{arm_id}/store.pt")
    mt = _load_store(cfg, f"stores/matched_tf/{arm_id}/store.pt")
    out: dict = {"arm": arm_id, "trees": {}, **_meta()}
    for tree, arm_store in (("onpolicy", own), ("matched_text", mt)):
        per: dict = {}
        for li in LAYERS:
            jtr = _joined_rows(arm_store, base, li, train_q)
            jev = _joined_rows(arm_store, base, li, eval_q)
            for k in LADDER_KS:
                bundle = torch.load(
                    cfg.out_root / "ladder" / f"prep_L{li}_k{k}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                pca, m0_prep, m0_Y = bundle["pca"], bundle["prep"], bundle["Y_train_v0"]
                Xp_tr = torch.as_tensor(_pca_apply(pca, jtr["cp"]), dtype=torch.float64)
                Yp_tr = torch.as_tensor(jtr["vp"], dtype=torch.float64)
                mp_prep = MA._ridge_prep(Xp_tr)
                c0_ev = torch.as_tensor(_pca_apply(pca, jev["c0"]), dtype=torch.float64)
                cp_ev = torch.as_tensor(_pca_apply(pca, jev["cp"]), dtype=torch.float64)
                vp_ev = jev["vp"]
                vp_tr_mean = jtr["vp"].mean(axis=0)
                preds = {
                    "rung1_M0_c0": MA._ridge_predict(m0_prep, m0_Y, c0_ev, lam_key="m0"),
                    "rung2_M0_cplus": MA._ridge_predict(m0_prep, m0_Y, cp_ev, lam_key="m0"),
                    "rung3_Mplus_c0": MA._ridge_predict(mp_prep, Yp_tr, c0_ev, lam_key="mp"),
                    "rung4_Mplus_cplus": MA._ridge_predict(mp_prep, Yp_tr, cp_ev, lam_key="mp"),
                }
                r2 = {
                    name: _pooled_r2(np.asarray(p, dtype=np.float64), vp_ev, vp_tr_mean)
                    for name, p in preds.items()
                }
                span = r2["rung4_Mplus_cplus"] - r2["rung1_M0_c0"]
                gap2 = (r2["rung2_M0_cplus"] - r2["rung1_M0_c0"]) / (span + 1e-12)
                gap3 = (r2["rung3_Mplus_c0"] - r2["rung1_M0_c0"]) / (span + 1e-12)
                interaction = (
                    (r2["rung4_Mplus_cplus"] - r2["rung1_M0_c0"])
                    - (r2["rung2_M0_cplus"] - r2["rung1_M0_c0"])
                    - (r2["rung3_Mplus_c0"] - r2["rung1_M0_c0"])
                )
                knn = {
                    "rung4_Mplus_cplus": knn_retrieval(
                        np.asarray(preds["rung4_Mplus_cplus"], dtype=np.float64), vp_ev
                    ),
                    "rung1_M0_c0": knn_retrieval(
                        np.asarray(preds["rung1_M0_c0"], dtype=np.float64), vp_ev
                    ),
                }
                per[f"L{li}_k{k}"] = {
                    "r2": r2,
                    "gap_closure_rung2": float(gap2),
                    "gap_closure_rung3": float(gap3),
                    "interaction": float(interaction),
                    "span_floor_to_ceiling": float(span),
                    "n_train": int(Xp_tr.shape[0]),
                    "n_eval": int(vp_ev.shape[0]),
                    "k_requested": int(k),
                    "k_real": int(pca["k_real"]),
                    "retained_var_frac": pca["retained_var_frac"],
                    "lambda_m0": bundle["lambda_m0"],
                    "lambda_mplus": float(mp_prep["_lam_memo"]["mp"][2]),
                    "knn": knn,
                    "identity_bias_baseline": "INAPPLICABLE — PCA-k inputs vs full-d outputs",
                }
        out["trees"][tree] = per
    dest = cfg.out_root / "ladder" / "arms" / f"{arm_id}.json"
    _atomic_json(dest, out)
    G._upload_paths(cfg.gcfg(), [dest], f"{HF_PREFIX}/ladder/arms")
    return [str(dest)]


def run_ladder_summary(cfg: Cfg, manifests: dict) -> list[str]:
    """Aggregate the per-arm ladder JSONs (four rungs, gap closures, interaction)."""
    arms = [r["arm_id"] for r in manifests["content_arms"]]
    per_arm = {}
    for a in arms:
        p = cfg.out_root / "ladder" / "arms" / f"{a}.json"
        if not p.exists():
            _ensure_local(cfg, f"ladder/arms/{a}.json")
        per_arm[a] = json.loads(p.read_text())
    rows = []
    for a, rec in per_arm.items():
        meta = manifests["arm_rows"][a]
        for tree, per in rec["trees"].items():
            for cell_key, cell in per.items():
                rows.append(
                    {
                        "arm": a,
                        "behavior": meta["behavior"],
                        "ctx": meta["ctx_key"],
                        "regime": meta["regime"],
                        "tree": tree,
                        "cell": cell_key,
                        **{
                            kk: cell[kk]
                            for kk in (
                                "r2",
                                "gap_closure_rung2",
                                "gap_closure_rung3",
                                "interaction",
                                "k_real",
                                "retained_var_frac",
                                "lambda_m0",
                                "lambda_mplus",
                                "n_train",
                                "n_eval",
                            )
                        },
                    }
                )
    dest = cfg.out_root / "ladder" / "summary.json"
    _atomic_json(dest, {"rows": rows, "n_arms": len(arms), **_meta()})
    G._upload_paths(cfg.gcfg(), [dest], f"{HF_PREFIX}/ladder")
    return [str(dest)]


# ── VM-side: harvest + figures ────────────────────────────────────────────────


def harvest(cfg: Cfg, manifests: dict) -> None:
    """Stage battery outputs from HF into eval_results/issue_1947/result3_theory/."""
    from explore_persona_space.orchestrate import hub

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{HF_PREFIX}/battery2/summary.json",
        EVAL_DIR / "summary.json",
        repo_type="dataset",
        overwrite=True,
    )
    (EVAL_DIR / "arms").mkdir(exist_ok=True)
    (EVAL_DIR / "ladder_arms").mkdir(exist_ok=True)
    hub.stage_hub_file(
        X.HF_DATA_REPO,
        f"{HF_PREFIX}/ladder/summary.json",
        EVAL_DIR / "ladder_summary.json",
        repo_type="dataset",
        overwrite=True,
    )
    for r in manifests["content_arms"]:
        a = r["arm_id"]
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{HF_PREFIX}/battery2/arms/{a}.json",
            EVAL_DIR / "arms" / f"{a}.json",
            repo_type="dataset",
            overwrite=True,
        )
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{HF_PREFIX}/ladder/arms/{a}.json",
            EVAL_DIR / "ladder_arms" / f"{a}.json",
            repo_type="dataset",
            overwrite=True,
        )
    logger.info(
        "[harvest] %d arm JSONs + summaries -> %s", len(manifests["content_arms"]), EVAL_DIR
    )


def cap_hit_report(cfg: Cfg, manifests: dict) -> None:
    """VM-side: realized cap-hit fraction per (state x panel family) from the
    uploaded raw generation shards; the >2% per-family re-gen trigger input."""
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate import hub

    panel = json.loads((cfg.config_dir / "prefix_panel.json").read_text())
    ext = json.loads((cfg.config_dir / "ext_members.json").read_text())["members"]
    fam_of = {m["prefix_id"]: m["family"] for m in panel["members"]}
    fam_of |= {m["prefix_id"]: EXT_FAMILY for m in ext}
    states = ["base_content"] + [r["arm_id"] for r in manifests["content_arms"]]
    out: dict = {"max_new_tokens": MAX_NEW_R3, "per_state": {}, **_meta()}
    fam_tot: dict[str, int] = {}
    fam_hit: dict[str, int] = {}
    for st in states:
        stage_dir = cfg.out_root / "caphit" / st
        rows: list[dict] = []
        for i in range(40):  # raw shards are <9 MB line-splits; probe until first miss
            rel = f"raw_completions/generation/{st}/{st}_generation.shard{i:02d}.jsonl"
            local = stage_dir / f"shard{i:02d}.jsonl"
            if not local.exists():
                try:
                    hub.stage_hub_file(
                        X.HF_DATA_REPO, f"{HF_PREFIX}/{rel}", local, repo_type="dataset"
                    )
                except EntryNotFoundError as exc:  # first missing index = end of shard set
                    if i == 0:
                        raise RuntimeError(f"no raw shards on HF for state {st}") from exc
                    break
            rows.extend(json.loads(line) for line in local.read_text().splitlines() if line.strip())
        n_hit = sum(1 for r in rows if r["finish_reason"] == "length")
        out["per_state"][st] = {"n_rows": len(rows), "n_cap_hit": n_hit}
        for r in rows:
            fam = fam_of.get(r["prefix_id"], "?")
            fam_tot[fam] = fam_tot.get(fam, 0) + 1
            if r["finish_reason"] == "length":
                fam_hit[fam] = fam_hit.get(fam, 0) + 1
    out["per_family"] = {
        f: {
            "n_rows": fam_tot[f],
            "n_cap_hit": fam_hit.get(f, 0),
            "frac": fam_hit.get(f, 0) / fam_tot[f],
            "over_2pct_trigger": (fam_hit.get(f, 0) / fam_tot[f]) > 0.02,
        }
        for f in sorted(fam_tot)
    }
    _atomic_json(EVAL_DIR / "cap_hit.json", out)
    logger.info(
        "[cap-hit] per-family: %s",
        {f: round(v["frac"], 4) for f, v in out["per_family"].items()},
    )


BEH_ORDER = ("sycophancy", "impolite", "writing_style")


def _save_fig(fig, name: str) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, name, dir=str(FIG_DIR))


def figures(cfg: Cfg) -> None:
    """Render the required figures from the harvested summary (VM-side)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    summary = json.loads((EVAL_DIR / "summary.json").read_text())
    colors = dict(zip(BEH_ORDER, paper_palette(3), strict=True))

    # 1) headline: CV error vs m (per-arm spaghetti + per-behavior mean), both trees
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, tree in zip(axes, ("onpolicy", "matched_text"), strict=True):
        rows = summary[tree]
        for beh in BEH_ORDER:
            curves = [r["cv_error_by_m"] for r in rows if r["behavior"] == beh]
            if not curves:
                continue
            m_axis = np.arange(len(curves[0]))
            for c in curves:
                ax.plot(m_axis, c, color=colors[beh], alpha=0.18, lw=0.8)
            ax.plot(m_axis, np.mean(curves, axis=0), color=colors[beh], lw=2.2, label=beh)
        floors = [r["noise_floor"] for r in rows]
        ax.axhline(float(np.median(floors)), ls="--", color="gray", lw=1.0)
        ax.set_xlabel("m (write directions, split-half CV)")
        ax.set_title({"onpolicy": "on-policy (primary)", "matched_text": "matched-text"}[tree])
    axes[0].set_ylabel("held-out reconstruction error (Fro ratio)")
    axes[0].legend(frameon=False)
    _save_fig(fig, "cv_error_vs_m")
    plt.close(fig)

    # 2) scalarity residual distributions (per arm, ECDF by behavior), both trees
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, tree in zip(axes, ("onpolicy", "matched_text"), strict=True):
        for r in summary[tree]:
            arm_rec = json.loads((EVAL_DIR / "arms" / f"{r['arm']}.json").read_text())
            vals = sorted(
                arm_rec["trees"][tree][f"L{PRIMARY_LAYER}"]["scalarity_residual"].values()
            )
            ax.plot(vals, np.linspace(0, 1, len(vals)), color=colors[r["behavior"]], alpha=0.35)
        ax.axvline(1.0, ls=":", color="gray", lw=1.0)
        ax.set_xlabel("scalarity residual  ||dv - w g|| / ||dv||")
        ax.set_title({"onpolicy": "on-policy (primary)", "matched_text": "matched-text"}[tree])
    axes[0].set_ylabel("ECDF over 49 target conditions")
    _save_fig(fig, "scalarity_residual_ecdf")
    plt.close(fig)

    # 3) six-way gate race (span-mean pooling, L19, psi = ridge-pinv headline;
    #    spearman per arm + perm null)
    combos = [
        ("c_C", "I"),
        ("c_C", "whitened"),
        ("psiPinv_t", "I"),
        ("psiPinv_t", "whitened"),
        ("c_C_plus_psiPinv_delta", "I"),
        ("c_C_plus_psiPinv_delta", "whitened"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = np.arange(len(combos))
    rngj = np.random.default_rng(0)
    for tree, marker, dx in (("onpolicy", "o", -0.12), ("matched_text", "s", 0.12)):
        for i, (k, m) in enumerate(combos):
            key = f"span_mean|{k}|{m}"
            for r in summary[tree]:
                g = r["gates"].get(key)
                if g is None:
                    continue
                ax.scatter(
                    i + dx + rngj.uniform(-0.05, 0.05),
                    g["spearman"],
                    s=14,
                    marker=marker,
                    color=colors[r["behavior"]],
                    alpha=0.55,
                )
    ax.axhline(0.0, color="gray", lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"k={k}\nM={m}" for k, m in combos], fontsize=8)
    ax.set_ylabel("Spearman(g_pred, g_realized) per arm")
    ax.set_title("Gate race: circles on-policy, squares matched-text")
    _save_fig(fig, "gate_race_sixway")
    plt.close(fig)

    # 4) rank grain comparison: top-1 share vs its n-matched null, both grains
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, tree in zip(axes, ("onpolicy", "matched_text"), strict=True):
        rows = summary[tree]
        for r in rows:
            ax.scatter(0, r["top1_uncentered"], color=colors[r["behavior"]], s=14, alpha=0.55)
            ax.scatter(1, r["top1_null_p97_5"], color="gray", s=8, alpha=0.4)
            ax.scatter(2, r["top1_prompt"], color=colors[r["behavior"]], s=14, alpha=0.55)
            if r["top1_prompt_null_p97_5"] is not None:
                ax.scatter(3, r["top1_prompt_null_p97_5"], color="gray", s=8, alpha=0.4)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(
            ["condition\n(real)", "condition\n(null p97.5)", "prompt\n(real)", "prompt\n(null)"],
            fontsize=8,
        )
        ax.set_title({"onpolicy": "on-policy (primary)", "matched_text": "matched-text"}[tree])
    axes[0].set_ylabel("top-1 variance share (uncentered)")
    _save_fig(fig, "rank_grains_vs_null")
    plt.close(fig)

    # 5) direction block: cos(w, delta) and cos(w, r_B) per arm vs null band
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for tree, marker, dx in (("onpolicy", "o", -0.1), ("matched_text", "s", 0.1)):
        for r in summary[tree]:
            ax.scatter(
                0 + dx,
                r["cos_w_delta"],
                marker=marker,
                s=14,
                color=colors[r["behavior"]],
                alpha=0.55,
            )
            ax.scatter(
                1 + dx,
                r["cos_w_rb"],
                marker=marker,
                s=14,
                color=colors[r["behavior"]],
                alpha=0.55,
            )
            ax.scatter(2 + dx, r["cov_null_abs_p95"], marker=marker, s=8, color="gray", alpha=0.4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["cos(w, delta)", "cos(w, r_B)", "corpus-cov null |cos| p95"])
    ax.axhline(0.0, color="gray", lw=1.0)
    ax.set_ylabel("cosine")
    ax.set_title("Write direction: circles on-policy, squares matched-text")
    _save_fig(fig, "direction_cosines")
    plt.close(fig)

    # 6) ablation ladder: four rungs faceted by tree, per-arm points behind bars
    lad = json.loads((EVAL_DIR / "ladder_summary.json").read_text())["rows"]
    rungs = ("rung1_M0_c0", "rung2_M0_cplus", "rung3_Mplus_c0", "rung4_Mplus_cplus")
    rung_labels = (
        "base map,\nbase context",
        "base map,\nfine-tuned context",
        "fine-tuned map,\nbase context",
        "fine-tuned map,\nfine-tuned context",
    )
    k_show = 512
    cell = f"L{PRIMARY_LAYER}_k{k_show}"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    rngj2 = np.random.default_rng(1)
    for ax, tree in zip(axes, ("onpolicy", "matched_text"), strict=True):
        rows = [r for r in lad if r["tree"] == tree and r["cell"] == cell]
        for i, rung in enumerate(rungs):
            vals = [r["r2"][rung] for r in rows]
            ax.bar(i, float(np.mean(vals)), width=0.6, color="lightgray", zorder=1)
            for r in rows:
                ax.scatter(
                    i + rngj2.uniform(-0.18, 0.18),
                    r["r2"][rung],
                    s=12,
                    color=colors[r["behavior"]],
                    alpha=0.55,
                    zorder=2,
                )
        ax.set_xticks(range(4))
        ax.set_xticklabels(rung_labels, fontsize=7)
        ax.set_title(
            {"onpolicy": "on-policy (primary)", "matched_text": "matched-text"}[tree]
            + f"  [L{PRIMARY_LAYER}, PCA k={k_show} inputs]"
        )
    axes[0].set_ylabel("held-out pooled R^2 vs measured fine-tuned profile")
    _save_fig(fig, "ablation_ladder")
    plt.close(fig)
    logger.info("[figures] wrote 6 figures -> %s", FIG_DIR)


# ── verification modes ────────────────────────────────────────────────────────


def import_check() -> None:
    """Resolve every deferred import + signature-bind the reused call sites."""
    import inspect

    import numpy as np  # noqa: F401
    import scipy.stats  # noqa: F401
    import torch  # noqa: F401

    import issue1768_capture as CAP
    import issue1768_directions as DIR
    import issue1947_cells as C47  # noqa: F401
    import issue1979_prep as PREP
    from explore_persona_space.analysis.paper_plots import (  # noqa: F401
        paper_palette,
        savefig_paper,
        set_paper_style,
    )
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _build_generation_prompts,
        _reap_vllm_engine,
        _teacher_forced_span_means,
        _vllm_enforce_eager,
        compute_prompt_spans,
    )
    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.preflight import (  # noqa: F401
        assert_out_root_headroom,
    )

    import issue825_map_alignment as MA
    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )

    binds = [
        (G.run_f1a, (None, None, "state")),
        (G.run_f1b_writes, (None, None, "arm", "base_content")),
        (G.run_f1c, (None, None, "mix")),
        (G.ensure_arm_registry, (None, None)),
        (G._upload_paths, (None, [], "dest")),
        (G._apply_saved_map, ({}, None, "cpu")),
        (CAP._mix_positive_rows, (None, None)),
        (DIR.corpus_sigma, (Path("/tmp"), 19)),
        (DIR.load_rb_tensors, (Path("/tmp"),)),
        (DIR.null_bands, (None, None, None)),
        (PREP._member, ("cid", "fam", "tier", "src", None, {})),
        (hub.stage_hub_file, ("repo", "rel", Path("/tmp/x"))),
        (X.pfx_resolve_context, ("cid",)),
        (MA._ridge_prep, (None,)),
        (MA._ridge_predict, (None, None, None)),
        (knn_retrieval, (None, None)),
    ]
    for fn, args in binds:
        inspect.signature(fn).bind(*args)
    print("[import-check] OK: deferred imports resolved + call sites signature-bound")


def plan_only(cfg: Cfg, manifests: dict, items: list[Item]) -> None:
    print(
        f"[plan] members={len(manifests['members'])} (targets={len(manifests['target_ids'])}) "
        f"queries={len(manifests['queries'])} arms={len(manifests['content_arms'])} "
        f"items={len(items)} sliced={cfg.limited}"
    )
    for it in items[:12]:
        print(f"  {it.key} deps={list(it.deps)}")


# ── main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--build-config", action="store_true")
    ap.add_argument("--panel-limit", type=int, default=None)
    ap.add_argument("--query-limit", type=int, default=None)
    ap.add_argument("--arms", default="")
    ap.add_argument("--smoke", action="store_true", help="smoke leg: tiny anchors + few nulls")
    ap.add_argument("--smoke-then-full", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--tf-batch", type=int, default=X.TF_BATCH_SIZE)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--null-cond", type=int, default=N_NULL_COND)
    ap.add_argument("--null-prompt", type=int, default=N_NULL_PROMPT)
    ap.add_argument("--perm-gate", type=int, default=N_PERM_GATE)
    ap.add_argument("--terminal-token", default="done", help=argparse.SUPPRESS)
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--cap-hit", action="store_true", help="VM-side per-family cap-hit report")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--worker-unit", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)

    if args.import_check:
        import_check()
        return 0
    out_root = (args.out_root or REPO_ROOT / "data" / "issue_1947" / "r3_theory").resolve()
    cfg = Cfg(
        out_root=out_root,
        config_dir=(args.config_dir or out_root / "config").resolve(),
        panel_limit=args.panel_limit,
        query_limit=args.query_limit,
        arms_filter=tuple(a for a in args.arms.split(",") if a),
        skip_upload=args.skip_upload,
        tf_batch=args.tf_batch,
        max_parallel=args.max_parallel,
        smoke=args.smoke,
        null_cond=args.null_cond,
        null_prompt=args.null_prompt,
        perm_gate=args.perm_gate,
        terminal_token=args.terminal_token,
    )
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    if args.build_config:
        build_config(cfg)
        return 0
    if args.smoke_then_full:
        # per-leg out-roots (crash-fix-rounds rule): the smoke leg never touches
        # production sentinels/resume state and never uploads.
        smoke_root = cfg.out_root.parent / (cfg.out_root.name + "-smoke")
        smoke_flags = [
            "--out-root",
            str(smoke_root),
            "--config-dir",
            str(cfg.config_dir),
            "--panel-limit",
            "6",
            "--query-limit",
            "2",
            "--arms",
            ",".join(SMOKE_ARMS),
            "--smoke",
            "--skip-upload",
            "--null-cond",
            "8",
            "--null-prompt",
            "4",
            "--perm-gate",
            "8",
        ]
        rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), *smoke_flags])
        if rc != 0:
            raise RuntimeError(f"smoke leg failed rc={rc} — full leg NOT started")
        print("[smoke-first] smoke leg passed — starting the full leg", flush=True)
    manifests = load_manifests(cfg)
    items = build_work_items(cfg, manifests)
    if args.plan_only:
        plan_only(cfg, manifests, items)
        return 0
    if args.harvest or args.figures or args.cap_hit:
        if args.harvest:
            harvest(cfg, manifests)
        if args.cap_hit:
            cap_hit_report(cfg, manifests)
        if args.figures:
            figures(cfg)
        return 0
    if args.worker_unit:
        t0 = time.time()
        from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

        try:
            assert_out_root_headroom(
                cfg.out_root, WORKER_HEADROOM_GB, phase=f"worker:{args.worker_unit}"
            )
            outputs = run_unit(cfg, manifests, args.worker_unit)
        except BaseException as exc:
            G._write_failure(cfg, args.worker_unit, exc)
            raise
        G._write_sentinel(cfg, args.worker_unit, time.time() - t0, outputs)
        G._failure_path(cfg, args.worker_unit).unlink(missing_ok=True)
        return 0
    dispatch(cfg, manifests, items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
