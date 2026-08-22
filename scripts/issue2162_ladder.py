#!/usr/bin/env python3
"""Issue #2162 follow-up ``persona-specificity-ladder`` — pod driver (plan v6 §4.6).

Forked THIN from ``scripts/issue2162_run.py`` (imported as ``RUN``): the same
phase/checkpoint/resume/claim-queue skeleton, and the parent's OWN
``_arm_hook_all_layers`` / ``capture_answer_states`` / ``run_injection_gate`` /
``margin_lnp`` / claim-queue / upload helpers are IMPORTED, never re-implemented.
Constants (temp 1.0, K=5 grid / K=10 anchor draws, 2048 cap, gate bars, seeds)
are the parent module's own pins — no fork-base constant flips.

Phases (plan §4.6 DAG; this driver owns P1/P2/P3/margin/P4):

- ``--phase bank`` (P1): stage the PARENT ``bank.json`` + ``vc_bank.pt`` from HF
  at the pinned revision (cross-type donor states + WildChat carrier texts;
  skip-if-present is the production resume predicate), freeze the ladder bank
  (7 values x 6 carriers, 12 directions x 72 pairs, both donor plans) into
  ``ladder_bank.json``, capture all-layer v_ce/v_pe for the 42 ladder contexts
  (right-padded forwards; positions off token ids — BPE-seam rule), run the
  DISTINCTNESS GUARD (plan §4.5: V(A) != V(B) at BOTH slots for every pair,
  cos < ``DEGENERACY_COS_MIN``; HALT ``RC_DEGENERACY_GATE``) and the inherited
  INJECTION-EXACTNESS GATE (12 ladder spot cells through the parent's
  parameterized ``run_injection_gate``; HALT ``RC_INJECTION_GATE``).
- ``--phase anchors`` (P2): 42 contexts x K=10 unpatched temp-1.0 rollouts
  (the anchor-separation gate wave inputs — floor AND ceiling per carrier),
  rollout text persisted BEFORE the V_a capture reduce, immediate upload so
  the VM judge starts while the pod idles for the verdict.
- ``--phase grid`` (P3): (direction x slot x arm) blocks over the GATE-SURVIVING
  rungs/carriers (``--gate-verdict`` JSON, judge-built; ``--donor-screen`` JSON
  picks each pair's screened cross-type donor) pulled from the parent's
  work-conserving claim-file queue; per block: K=5 hooked temp-1.0 draws per
  pair, the hooked teacher-forced V_a pass, inline margin TF when ``--pools``
  is present; per-block JSONL/pt checkpoints + incremental upload.
- ``--phase margin``: pools-dependent TF legs (anchor margins + the per-block
  catch-up, claim-queue namespace ``margin_blocks``).
- ``--phase upload`` (P4): ONE bulk ``upload_folder`` commit per HF prefix
  under ``issue2162_ctxinfo/{raw_completions,analysis_tensors}/ladder/``, then
  the pod sentinel ``/workspace/logs/issue-2162-results.json``.

``--smoke`` slices the BANK GRID ONLY (plan §4.6: R1 x d1 x both slots x 3 arms
x 1 draw + anchors K=2 on the smoke contexts) — no implementation
substitutions, no gate downgrades; the production gates run identically.

Pod-side contract: sentinel file + ``[phase=...]`` breadcrumbs ONLY — this file
NEVER shells out to ``scripts/task.py``. Every phase ends with an explicit
``sys.exit`` (#1689 finalization-race rule).
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import torch  # noqa: E402

import issue2162_run as RUN  # noqa: E402  (same-dir script import, parent skeleton)
from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)
from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    generate_batch,
)
from explore_persona_space.experiments.issue2094 import bank as BANK94  # noqa: E402
from explore_persona_space.experiments.issue2094.fmetrics import safe_cosine  # noqa: E402
from explore_persona_space.experiments.issue2162 import ladder_bank as LB  # noqa: E402

logger = logging.getLogger("issue2162.ladder")

# ── round constants (plan v6 §4/§9/§10; everything else = RUN's own pins) ──

DEFAULT_OUT_ROOT = Path("/workspace/issue2162_out/ladder")
HF_LADDER_RAW = f"{RUN.HF_PREFIX}/raw_completions/ladder"
HF_LADDER_TENSORS = f"{RUN.HF_PREFIX}/analysis_tensors/ladder"
PARENT_VC_BANK_PREFIX = f"{RUN.HF_PREFIX}/analysis_tensors/vc_bank"

ARMS_LADDER: tuple[str, ...] = ("steered", "null_sameval", "null_xtype")
SMOKE_GRID_DRAWS = 1  # plan §4.6 smoke: 1 draw (parent smoke used 2)
SMOKE_ANCHOR_DRAWS = 2
SMOKE_DIRECTIONS: tuple[str, ...] = ("install_r1_pirate", "erase_r1_pirate")
SMOKE_CARRIER = "d1"


# ── config ────────────────────────────────────────────────────────────


@dataclass
class LadderConfig(RUN.RunConfig):
    """Parent RunConfig + the ladder round's own knobs (defaulted, appended)."""

    parent_dir: Path = DEFAULT_OUT_ROOT / "parent_bank"
    hf_revision: str = LB.PARENT_HF_REVISION
    gate_verdict_path: Path | None = None
    donor_screen_path: Path | None = None

    @property
    def rollouts_dir(self) -> Path:  # plan §9 phase_outputs: grid/*.jsonl
        return self.out_root / "grid"

    @property
    def gates_dir(self) -> Path:
        return self.out_root / "gates"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2162 ladder pod driver (bank / anchors / grid / margin / upload)."
    )
    ap.add_argument(
        "--phase",
        choices=("bank", "anchors", "grid", "margin", "upload"),
        help="pipeline phase to run (required unless --import-check)",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="resolve every deferred import (incl. function-body imports) + the "
        "argparse-attribute completeness assert, then exit 0",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--log-dir", type=Path, default=RUN.DEFAULT_LOG_DIR)
    ap.add_argument("--model-id", default=RUN.MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="from-config tiny CPU model (smoke)")
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    ap.add_argument("--gen-batch", type=int, default=16, help="cells per hooked generate call")
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=RUN.MAX_NEW_TOKENS)
    ap.add_argument("--anchor-draws", type=int, default=RUN.ANCHOR_DRAWS)
    ap.add_argument("--grid-draws", type=int, default=RUN.GRID_DRAWS)
    ap.add_argument("--seed-base", type=int, default=RUN.SEED_BASE)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="R1 x d1 x both slots x 3 arms x 1 draw + anchors K=2 (bank-grid slice ONLY)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-run a completed phase (resume override ONLY — never bypasses HALT gates)",
    )
    ap.add_argument(
        "--force-past-halt-gates",
        action="store_true",
        help="DANGEROUS: proceed past a FAILED HALT gate (manual diagnosis only)",
    )
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=None, help="informational; CVD pins the device")
    ap.add_argument("--upload", choices=("hf", "local-mirror", "none"), default="hf")
    ap.add_argument(
        "--upload-every",
        type=int,
        default=25,
        help="grid: bulk-upload staged text every N blocks (256 commits/hr cap)",
    )
    ap.add_argument(
        "--pools",
        type=Path,
        default=None,
        help="ladder margin pools JSON (judge-built, keys = direction ids)",
    )
    ap.add_argument(
        "--parent-dir",
        type=Path,
        default=None,
        help="staged parent-bank dir holding bank.json + vc_bank.pt "
        "(default <out-root>/parent_bank; pre-staged files skip the HF fetch — "
        "the production resume predicate, also the tiny-fixture seam)",
    )
    ap.add_argument(
        "--hf-revision",
        default=LB.PARENT_HF_REVISION,
        help="parent-artifact revision pin on the data repo",
    )
    ap.add_argument(
        "--gate-verdict",
        type=Path,
        default=None,
        help="anchor-separation gate verdict JSON (judge-built); REQUIRED for a "
        "non-smoke grid — the gate runs BEFORE any grid spend (plan §4.5)",
    )
    ap.add_argument(
        "--donor-screen",
        type=Path,
        default=None,
        help="cross-type donor screen JSON (judge-built); REQUIRED for a non-smoke "
        "grid (smoke uses the frozen PRIMARY donors unscreened — declared blind spot)",
    )
    ap.add_argument("--planned-wall-h", type=float, default=1.4)  # plan §9 P3 row
    ap.add_argument("--gpu-hours-budgeted", type=float, default=3.0)  # plan §9 total
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> LadderConfig:
    if args.device:
        device = args.device
    elif args.tiny:
        device = "cpu"
    else:
        device = "cuda:0"
    return LadderConfig(
        phase=args.phase,
        out_root=args.out_root,
        log_dir=args.log_dir,
        model_id=args.model_id,
        tiny=args.tiny,
        n_layers=args.tiny_layers if args.tiny else RUN.N_MODEL_LAYERS_FULL,
        hidden=args.tiny_hidden if args.tiny else RUN.HIDDEN_FULL,
        device=device,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        max_new_tokens=args.max_new_tokens,
        anchor_draws=args.anchor_draws,
        grid_draws=args.grid_draws,
        seed_base=args.seed_base,
        smoke=args.smoke,
        pilot=False,  # no pilot leg this round (plan §7 lists 3 gates; parent basis MEASURED)
        force=args.force,
        force_past_halt_gates=args.force_past_halt_gates,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
        upload_mode=args.upload,
        upload_every=args.upload_every,
        planned_wall_h=args.planned_wall_h,
        gpu_hours_budgeted=args.gpu_hours_budgeted,
        pools_path=args.pools,
        parent_dir=args.parent_dir
        if args.parent_dir is not None
        else args.out_root / "parent_bank",
        hf_revision=args.hf_revision,
        gate_verdict_path=args.gate_verdict,
        donor_screen_path=args.donor_screen,
    )


# ── parent-bank staging (plan §4.6 / artifact-reuse (h)) ──────────────

PARENT_FILES = ("bank.json", "vc_bank.pt")


def stage_parent_bank(cfg: LadderConfig) -> None:
    """Stage the parent ``bank.json`` + ``vc_bank.pt`` at the pin.

    Consumer opens the EXACT fetch destinations (no staging transformation —
    leg (h)(iv) N/A escape). Skip-if-present is the production resume
    predicate (a pre-staged ``--parent-dir`` — pod resume, or the tiny-fixture
    seam — takes the same skip branch the production resume takes). On the
    DOWNLOAD path the pairwise-provenance ordering (plan item (j): bank.json
    last-commit <= vc_bank.pt last-commit at the pin) is asserted first.
    """
    missing = [name for name in PARENT_FILES if not (cfg.parent_dir / name).exists()]
    if not missing:
        logger.info("[stage] parent bank present under %s — skipping fetch", cfg.parent_dir)
        return
    from huggingface_hub import HfApi, hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    paths = [f"{PARENT_VC_BANK_PREFIX}/{name}" for name in PARENT_FILES]
    infos = retry_transient(
        lambda: HfApi().get_paths_info(
            RUN.HF_DATA_REPO, paths, expand=True, repo_type="dataset", revision=cfg.hf_revision
        ),
        what="parent-bank get_paths_info",
    )
    by_path = {i.path: i for i in infos}
    missing_remote = [p for p in paths if p not in by_path]
    assert not missing_remote, (
        f"parent bank files missing at pin {cfg.hf_revision}: {missing_remote}"
    )
    dates = {p: by_path[p].last_commit.date for p in paths}
    assert dates[paths[0]] <= dates[paths[1]], (
        f"pairwise provenance violated: bank.json ({dates[paths[0]]}) postdates "
        f"vc_bank.pt ({dates[paths[1]]}) at the pin — re-pin per artifact-reuse (j)"
    )
    cfg.parent_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        target = cfg.parent_dir / name
        with tempfile.TemporaryDirectory(dir=cfg.parent_dir, prefix=".hfstage_") as td:
            got = retry_transient(
                lambda name=name, td=td: hf_hub_download(
                    RUN.HF_DATA_REPO,
                    f"{PARENT_VC_BANK_PREFIX}/{name}",
                    repo_type="dataset",
                    revision=cfg.hf_revision,
                    local_dir=td,
                ),
                what=f"parent-bank fetch {name}",
            )
            os.replace(got, target)  # same filesystem (td lives inside parent_dir)
        logger.info("[stage] fetched %s @ %s", target, cfg.hf_revision)


def load_parent_bank(cfg: LadderConfig) -> tuple[dict, dict]:
    """``(parent bank.json manifest, parent per-context state records)``."""
    stage_parent_bank(cfg)
    manifest = json.loads((cfg.parent_dir / "bank.json").read_text())
    # Self-produced, sha-recorded bundle carrying non-tensor metadata (parent
    # convention — torch>=2.6 weights_only=False for the pinned self-produced .pt).
    bundle = torch.load(cfg.parent_dir / "vc_bank.pt", map_location="cpu", weights_only=False)
    recs = bundle["per_context"]
    assert isinstance(recs, dict) and recs, "parent vc_bank.pt carries no per_context records"
    return manifest, recs


# ── ladder bank manifest / regime / blocks ────────────────────────────


def ladder_manifest_and_sha(parent_manifest: dict) -> tuple[dict, str]:
    manifest = LB.ladder_bank_manifest(parent_manifest, LB.SEED)
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    return manifest, RUN._sha256_bytes(payload)


def _load_ladder_manifest(cfg: LadderConfig) -> tuple[dict, str]:
    path = cfg.bank_dir / "ladder_bank.json"
    assert path.exists(), f"{path} missing — run `--phase bank` first"
    manifest = json.loads(path.read_text())
    sha = manifest.get("bank_sha")
    assert isinstance(sha, str) and sha, "ladder_bank.json carries no bank_sha"
    return manifest, sha


def pairs_from_manifest(manifest: dict) -> list[LB.LadderPair]:
    """Frozen pair rows -> LadderPair structs (the registry the queue runs on)."""
    pairs = [
        LB.LadderPair(
            pair_id=row["pair_id"],
            cell=row["direction"],
            kind=row["kind"],
            persona=row["persona"],
            carrier=row["carrier"],
            value_a=row["value_a"],
            value_b=row["value_b"],
            a=row["a"],
            b=row["b"],
        )
        for row in manifest["pairs"]
    ]
    assert pairs, "ladder manifest carries no pairs"
    return pairs


def read_gate_verdict(path: Path) -> dict[str, list[str]]:
    """``persona value_id -> sorted surviving carriers`` from the judge-built
    verdict JSON (plan §7 gate 1). ALL-rungs-fail was already a judge-side
    HALT; an empty survivor map here still fails loud."""
    payload = json.loads(path.read_text())
    rungs = payload.get("rungs")
    assert isinstance(rungs, dict) and rungs, f"{path} carries no 'rungs' object"
    survivors: dict[str, list[str]] = {}
    for value_id, rec in rungs.items():
        assert value_id in LB.PERSONA_VALUE_IDS, (value_id, "unknown rung in gate verdict")
        if rec.get("survived"):
            carriers = rec.get("surviving_carriers")
            assert isinstance(carriers, list) and carriers, (value_id, "survived with no carriers")
            survivors[value_id] = sorted(carriers)
    assert survivors, (
        f"{path}: NO surviving rungs — the judge-side gate should have HALTed "
        "(rig-defect branch); refusing to run an empty grid"
    )
    return survivors


def read_donor_screen(path: Path) -> dict[str, dict]:
    """``pair_id -> {"status", "donor"}`` from the judge-built screen JSON."""
    payload = json.loads(path.read_text())
    assignments = payload.get("assignments")
    assert isinstance(assignments, dict) and assignments, f"{path} carries no 'assignments'"
    for pid, rec in assignments.items():
        status = rec.get("status")
        assert status in ("primary", "alternate", "dropped"), (pid, status)
        if status != "dropped":
            donor = rec.get("donor")
            assert isinstance(donor, dict) and donor.get("b"), (pid, "qualified with no donor row")
    return assignments


def donor_maps_ladder(
    manifest: dict,
    pairs: list[LB.LadderPair],
    survivors: dict[str, list[str]] | None,
    screen: dict[str, dict] | None,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Both null-arm donor maps for the given survivor set.

    - ``null_sameval``: pair_id -> LADDER donor context id (recipient's SOURCE
      value, next gate-surviving carrier in the frozen cyclic order).
    - ``null_xtype``: pair_id -> PARENT donor B-context id (screened choice
      when ``screen`` is given, else the frozen PRIMARY).

    Returns ``(donor_maps, dropped_xtype_pair_ids)`` — a screen-dropped pair
    runs steered + null_sameval but NOT null_xtype (drop + report, plan §4.2).
    """
    order = tuple(manifest["sameval_donor"]["order"])
    plan = manifest["crosstype_donor_plan"]
    sameval: dict[str, str] = {}
    xtype: dict[str, str] = {}
    dropped: list[str] = []
    for pair in pairs:
        surv = (
            survivors.get(pair.persona, [])
            if survivors is not None
            else list(LB.carrier_ids(LB.SEED))
        )
        if pair.carrier in surv:
            donor_carrier = LB.sameval_donor_carrier(pair.carrier, surv, order)
            sameval[pair.pair_id] = LB.context_id(pair.value_a, donor_carrier)
        if screen is not None:
            rec = screen.get(pair.pair_id)
            assert rec is not None, (pair.pair_id, "missing from donor screen")
            if rec["status"] == "dropped":
                dropped.append(pair.pair_id)
            else:
                xtype[pair.pair_id] = rec["donor"]["b"]
        else:
            xtype[pair.pair_id] = plan[pair.pair_id]["primary"]["b"]
    return {"null_sameval": sameval, "null_xtype": xtype}, dropped


def enumerate_ladder_blocks(
    pairs: list[LB.LadderPair],
    survivors: dict[str, list[str]] | None,
    dropped_xtype: set[str] | None = None,
) -> list[RUN.Block]:
    """(direction x slot x arm) blocks over the surviving rungs/carriers.

    ``survivors=None`` keeps every rung/carrier (the pre-gate ceiling: 72
    blocks). A gate-dropped rung/carrier generates NOTHING (plan §4.3); a
    screen-dropped pair is excluded from its ``null_xtype`` blocks only."""
    dropped_xtype = dropped_xtype or set()
    by_direction: dict[str, list[LB.LadderPair]] = {}
    for p in pairs:
        by_direction.setdefault(p.cell, []).append(p)
    blocks: list[RUN.Block] = []
    for direction in LB.direction_ids():
        dpairs = sorted(by_direction[direction], key=lambda p: p.carrier)
        persona = dpairs[0].persona
        if survivors is not None:
            if persona not in survivors:
                continue
            dpairs = [p for p in dpairs if p.carrier in survivors[persona]]
        if not dpairs:
            continue
        for slot in RUN.SLOTS:
            for arm in ARMS_LADDER:
                ids = tuple(
                    p.pair_id
                    for p in dpairs
                    if not (arm == "null_xtype" and p.pair_id in dropped_xtype)
                )
                if not ids:
                    continue
                blocks.append(RUN.Block(direction, slot, arm, ids))
    keys = [b.key for b in blocks]
    assert len(set(keys)) == len(keys), "duplicate ladder block keys"
    return blocks


def smoke_ladder_blocks(pairs: list[LB.LadderPair]) -> list[RUN.Block]:
    """Plan §4.6 smoke slice: R1 x d1 x both slots x 3 arms (1 pair each)."""
    by_id = {(p.cell, p.carrier): p for p in pairs}
    blocks: list[RUN.Block] = []
    for direction in SMOKE_DIRECTIONS:
        pair = by_id[(direction, SMOKE_CARRIER)]
        for slot in RUN.SLOTS:
            for arm in ARMS_LADDER:
                blocks.append(RUN.Block(direction, slot, arm, (pair.pair_id,)))
    assert len(blocks) == 12, len(blocks)
    return blocks


def smoke_context_ids(pairs: list[LB.LadderPair]) -> list[str]:
    ids: list[str] = []
    by_id = {p.pair_id: p for p in pairs}
    for block in smoke_ladder_blocks(pairs):
        for pid in block.pair_ids:
            for cid in (by_id[pid].a, by_id[pid].b):
                if cid not in ids:
                    ids.append(cid)
    return ids


# ── payloads (parent geometry; ladder + parent donor states) ──────────


def payload_for_arm_ladder(
    bank: dict,
    pair: LB.LadderPair,
    slot: str,
    arm: str,
    donor_maps: dict[str, dict[str, str]],
    pairs_by_id: dict[str, LB.LadderPair],
    *,
    parent_recs: dict | None = None,
) -> tuple[torch.Tensor, str | None]:
    """``((1, L, H) payload, donor_context_id)`` for one (pair, slot, arm).

    - ``steered``: the pair's OWN target-context state V_slot(B) — raw, the
      parent steered convention (no norm matching).
    - ``null_sameval``: V_slot of the recipient's SOURCE value under the
      frozen donor carrier (a LADDER context), norm-matched per layer to the
      recipient pair's V(target) (plan §4.2 arm 2).
    - ``null_xtype``: V_slot(B) of the screened PARENT cross-type donor pair
      (``instr_format``/``verbosity``), norm-matched per layer (arm 3).
    """
    del pairs_by_id  # signature parity with RUN.payload_for_arm (gate seam)
    recs = bank["per_context"]
    recipient = RUN._slot_state(recs[pair.b], slot).unsqueeze(0)  # (1, L, H)
    if arm == "steered":
        return recipient.clone(), None
    donor_cid = donor_maps[arm][pair.pair_id]
    if arm == "null_sameval":
        donor_rec = recs[donor_cid]
    else:
        assert arm == "null_xtype", arm
        assert parent_recs is not None, "null_xtype payload needs the staged parent records"
        donor_rec = parent_recs.get(donor_cid)
        assert donor_rec is not None, (
            f"cross-type donor context {donor_cid!r} missing from the staged parent "
            "vc_bank (plan §12 assumption 2/8 — fail loud, never a silent fallback)"
        )
    donor_state = RUN._slot_state(donor_rec, slot).unsqueeze(0)
    assert donor_state.shape == recipient.shape, (donor_state.shape, recipient.shape)
    return BANK94.norm_match(donor_state, recipient), donor_cid


def _ladder_gate_spots(pairs: list[LB.LadderPair]) -> list[dict]:
    """12 injection-gate spot cells spanning all 12 directions, both slots,
    all 3 arms, and hand-written + WildChat carriers (plan §4.5 gate shape)."""
    by_key = {(p.cell, p.carrier): p for p in pairs}
    spec: list[tuple[str, str, str, str]] = [
        ("install_r1_pirate", "ce", "steered", "d1"),
        ("erase_r1_pirate", "pe", "null_sameval", "n3"),
        ("install_r2_butler", "ce", "null_xtype", "d2"),
        ("erase_r2_butler", "pe", "steered", "n4"),
        ("install_r3_warm", "ce", "null_sameval", "n7"),
        ("erase_r3_warm", "ce", "steered", "d1"),
        ("install_r4_trait", "pe", "steered", "n9"),
        ("erase_r4_trait", "ce", "null_xtype", "d2"),
        ("install_r5a_lu_therapy", "ce", "steered", "n3"),
        ("erase_r5a_lu_therapy", "pe", "null_xtype", "n7"),
        ("install_r5b_lu_philosophy", "pe", "null_sameval", "d2"),
        ("erase_r5b_lu_philosophy", "ce", "steered", "n9"),
    ]
    out = [
        {"cell": d, "slot": slot, "arm": arm, "pair": by_key[(d, carrier)]}
        for d, slot, arm, carrier in spec
    ]
    assert len(out) == 12, len(out)
    assert {s["cell"] for s in out} == set(LB.direction_ids()), "gate spots miss a direction"
    return out


# ── P1: ladder bank capture + gates ───────────────────────────────────


@torch.no_grad()
def capture_ladder_bank(cfg: LadderConfig, model, tok, contexts: dict[str, dict]) -> dict:
    """All-layer v_ce + v_pe per ladder context (parent capture geometry:
    right-padded forwards, positions off token ids — BPE-seam rule)."""
    ctx_ids = {cid: BANK94.context_token_ids_2094(tok, c) for cid, c in contexts.items()}
    prefix_ends = {cid: BANK94.prefix_end_index_multi(tok, ids) for cid, ids in ctx_ids.items()}
    layers = cfg.layers
    pad_id = tok.pad_token_id
    records: dict[str, dict] = {}
    order = list(contexts)
    for start in range(0, len(order), cfg.capture_batch):
        chunk = order[start : start + cfg.capture_batch]
        ids, mask = RUN._right_pad([ctx_ids[c] for c in chunk], pad_id, cfg.device)
        captured = extract_layer_activations(model, ids, layers, attention_mask=mask)
        for j, cid in enumerate(chunk):
            ctx_len = len(ctx_ids[cid])
            pe = prefix_ends[cid]
            assert 1 <= pe < ctx_len, (cid, ctx_len, pe)
            v_ce = torch.stack([captured[layer][j, ctx_len - 1] for layer in layers])
            v_pe = torch.stack([captured[layer][j, pe - 1] for layer in layers])
            assert v_ce.shape == (len(layers), cfg.hidden), v_ce.shape
            ctx = contexts[cid]
            records[cid] = {
                "context_id": cid,
                "cell": ctx["cell"],
                "value_id": ctx["value_id"],
                "rung": ctx["rung"],
                "carrier": ctx["carrier"],
                "ctx_len": ctx_len,
                "prefix_end": pe,
                "v_ce": v_ce.float().cpu(),
                "v_pe": v_pe.float().cpu(),
            }
        del captured
        logger.info(
            "[bank] unit %d/%d contexts elapsed",
            min(start + cfg.capture_batch, len(order)),
            len(order),
        )
    assert len(records) == len(contexts), (len(records), len(contexts))
    return {"layers": layers, "per_context": records}


def run_distinctness_guard(bank: dict, pairs: list[LB.LadderPair]) -> dict:
    """Plan §4.5 distinctness guard: V(A) != V(B) at BOTH slots for EVERY pair
    (cos < ``DEGENERACY_COS_MIN``). No pre-declared degenerate cells exist in
    this bank; a violation is a bank defect -> HALT."""
    recs = bank["per_context"]
    violations: list[dict] = []
    for pair in pairs:
        ra, rb = recs[pair.a], recs[pair.b]
        for slot in RUN.SLOTS:
            cos = float(
                safe_cosine(
                    RUN._slot_state(ra, slot).flatten(), RUN._slot_state(rb, slot).flatten()
                )
            )
            if not (cos < RUN.DEGENERACY_COS_MIN):
                violations.append({"pair_id": pair.pair_id, "slot": slot, "cos": cos})
    return {
        "criterion": "ladder distinctness guard (plan §4.5): V(A) != V(B) at both slots",
        "bar_cos": RUN.DEGENERACY_COS_MIN,
        "n_pairs_checked": len(pairs),
        "n_violations": len(violations),
        "violations": violations[:50],
        "passed": not violations,
    }


def bank_is_done_ladder(cfg: LadderConfig, regime_fp: str) -> bool:
    rec = RUN._phase_done_record(cfg, "bank", regime_fp)
    if rec is None:
        return False
    required = [
        cfg.bank_dir / "ladder_bank.json",
        cfg.bank_dir / "vc_bank_ladder.pt",
        cfg.gates_dir / "injection_gate_report.json",
        cfg.gates_dir / "distinctness_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        logger.warning(
            "[bank] done-manifest present but artifacts missing %s — re-running", missing
        )
        return False
    return True


def phase_bank(cfg: LadderConfig) -> int:
    """P1: parent staging + ladder_bank.json + vc_bank_ladder.pt + both gates."""
    logger.info("[phase=bank]")
    parent_manifest, parent_recs = load_parent_bank(cfg)
    manifest, bank_sha = ladder_manifest_and_sha(parent_manifest)
    regime_fp = RUN.regime_fingerprint(cfg, bank_sha)
    if not cfg.force and bank_is_done_ladder(cfg, regime_fp):
        logger.info("[bank] already done for this regime — skipping (--force re-runs)")
        logger.info("[phase=bank_done]")
        return RUN.RC_OK

    pairs = pairs_from_manifest(manifest)
    # Plan §12 assumption 2/8 structural probe, FULL consumed grain: every
    # planned cross-type donor context (primary AND alternate) has a state row
    # in the staged parent vc_bank — fail loud BEFORE any capture spend.
    plan_rows = manifest["crosstype_donor_plan"]
    missing_donors = sorted(
        {
            row[role]["b"]
            for row in plan_rows.values()
            for role in ("primary", "alternate")
            if row[role]["b"] not in parent_recs
        }
    )
    assert not missing_donors, (
        f"{len(missing_donors)} planned cross-type donor contexts missing from the "
        f"staged parent vc_bank (first: {missing_donors[:5]}) — plan §12 assumption 2/8"
    )
    logger.info(
        "[probe] parent grain OK: %d contexts staged; %d/%d donor plans resolved",
        len(parent_recs),
        len(plan_rows),
        len(pairs),
    )

    model, tok = RUN.load_model_and_tokenizer(cfg)
    # Plan §4.1 / §12 assumption 4: explicit plain block vs omitted-system render.
    equality = LB.plain_render_equality(tok)
    if not equality["equal"]:
        logger.warning(
            "[bank] plain-render equality FAILED (token_delta=%s) — keeping the explicit "
            "block, recording the comparability caveat (plan §4.1)",
            equality["token_delta"],
        )
    manifest["plain_render_equality"] = equality
    manifest["bank_sha"] = bank_sha
    manifest["repro"] = RUN._repro(cfg)
    RUN._write_json_atomic(cfg.bank_dir / "ladder_bank.json", manifest)

    contexts = manifest["contexts"]
    bank = capture_ladder_bank(cfg, model, tok, contexts)
    RUN._save_pt_atomic(
        cfg.bank_dir / "vc_bank_ladder.pt",
        {
            "layers": bank["layers"],
            "per_context": bank["per_context"],
            "bank_sha": bank_sha,
            "repro": RUN._repro(cfg),
        },
    )
    logger.info("[bank] captured %d contexts x %d layers", len(bank["per_context"]), cfg.n_layers)

    distinctness = run_distinctness_guard(bank, pairs)
    distinctness["repro"] = RUN._repro(cfg)
    RUN._write_json_atomic(cfg.gates_dir / "distinctness_report.json", distinctness)
    if not distinctness["passed"]:
        logger.error(
            "[distinctness_guard] FAILED: %d/%d pair-slot violations (bank defect)",
            distinctness["n_violations"],
            distinctness["n_pairs_checked"],
        )
        if not cfg.force_past_halt_gates:
            return RUN.RC_DEGENERACY_GATE
        logger.error("[distinctness_guard] --force-past-halt-gates set: proceeding (recorded)")

    # Injection gate over the frozen PRIMARY donors (screen runs judge-side
    # later; exactness is donor-choice-independent).
    donor_maps, _dropped = donor_maps_ladder(manifest, pairs, survivors=None, screen=None)
    report = RUN.run_injection_gate(
        cfg,
        model,
        tok,
        bank,
        pairs,
        donor_maps,
        contexts=contexts,
        ids_fn=BANK94.context_token_ids_2094,
        spots=_ladder_gate_spots(pairs),
        payload_fn=functools.partial(payload_for_arm_ladder, parent_recs=parent_recs),
    )
    RUN._write_json_atomic(cfg.gates_dir / "injection_gate_report.json", report)
    if not report["passed"]:
        logger.error(
            "[injection_gate] FAILED: %d/%d spots failed",
            report["n_spots_failed"],
            report["n_spots"],
        )
        if not cfg.force_past_halt_gates:
            return RUN.RC_INJECTION_GATE
        logger.error("[injection_gate] --force-past-halt-gates set: proceeding (recorded)")

    RUN._write_json_atomic(
        cfg.manifest_dir / "bank_done.json",
        {
            "regime_fp": regime_fp,
            "bank_sha": bank_sha,
            "n_contexts": len(bank["per_context"]),
            "plain_render_equal": bool(equality["equal"]),
            "injection_gate_passed": bool(report["passed"]),
            "distinctness_gate_passed": bool(distinctness["passed"]),
            "forced_past_gate": bool(
                cfg.force_past_halt_gates and not (report["passed"] and distinctness["passed"])
            ),
            "repro": RUN._repro(cfg),
        },
    )
    # Immediate upload (persist-by-default; P4 re-covers the same paths): the
    # VM judge's donor screen stages the FROZEN ladder_bank.json (donor plan)
    # from the Hub BEFORE P4's terminal upload exists.
    RUN._upload_dir(cfg, cfg.bank_dir, f"{HF_LADDER_TENSORS}/vc_bank", ["*.json", "*.pt"])
    RUN._upload_dir(cfg, cfg.gates_dir, f"{HF_LADDER_TENSORS}/gates", ["*.json"])
    logger.info("[phase=bank_done]")
    return RUN.RC_OK


# ── P2: anchors (all contexts, unpatched; the gate-wave inputs) ───────


def phase_anchors(cfg: LadderConfig) -> int:
    """P2: 42 contexts x K=10 unpatched temp-1.0 rollouts + V_a capture; text
    persisted BEFORE the capture reduce; immediate upload for the VM judge."""
    logger.info(
        "[phase=anchors] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    manifest, bank_sha = _load_ladder_manifest(cfg)
    regime_fp = RUN.regime_fingerprint(cfg, bank_sha)
    draws = SMOKE_ANCHOR_DRAWS if cfg.smoke else cfg.anchor_draws
    contexts: dict[str, dict] = manifest["contexts"]
    order = list(contexts)
    if cfg.smoke:
        keep = set(smoke_context_ids(pairs_from_manifest(manifest)))
        order = [cid for cid in order if cid in keep]
    my_order = order[cfg.worker_index :: cfg.num_workers]

    if not cfg.force and RUN._anchor_batch_done(cfg, regime_fp, "gate", draws):
        logger.info("[anchors] already done for this regime — skipping")
        logger.info("[phase=anchors_done] worker=%d", cfg.worker_index)
        return RUN.RC_OK

    model, tok = RUN.load_model_and_tokenizer(cfg)
    eot = RUN.eot_tail_ids(tok)
    ctx_ids = {cid: BANK94.context_token_ids_2094(tok, contexts[cid]) for cid in my_order}
    rows: list[dict] = []
    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    t0 = time.monotonic()
    for start in range(0, len(my_order), cfg.gen_batch):
        chunk = my_order[start : start + cfg.gen_batch]
        outs = generate_batch(
            model,
            tok,
            [contexts[c] for c in chunk],
            n=draws,
            hook=None,
            max_new_tokens=cfg.max_new_tokens,
            temperature=RUN.ANCHOR_TEMPERATURE,
            seed_base=cfg.seed_base,
            render_fn=BANK94.render_context_2094,
            ids_fn=BANK94.context_token_ids_2094,
        )
        for b, cid in enumerate(chunk):
            ctx = contexts[cid]
            for i, text in enumerate(outs[b]):
                flat_ctx.append(ctx_ids[cid])
                flat_text.append(text)
                rows.append(
                    {
                        "context_id": cid,
                        "cell": ctx["cell"],
                        "value_id": ctx["value_id"],
                        "rung": ctx["rung"],
                        "carrier": ctx["carrier"],
                        "draw": i,
                        "seed": cfg.seed_base + i,
                        "temperature": RUN.ANCHOR_TEMPERATURE,
                        "gate_slice": True,  # every ladder anchor feeds the gate wave
                        "text": text,
                    }
                )
        logger.info(
            "[anchors] unit %d/%d contexts elapsed=%.1fs",
            min(start + cfg.gen_batch, len(my_order)),
            len(my_order),
            time.monotonic() - t0,
        )
    # Persist rollout TEXT the moment generation completes, BEFORE the capture
    # reduce (#779); the post-capture write atomically REPLACES it enriched.
    jsonl = cfg.anchors_dir / f"anchors_gate_w{cfg.worker_index}.jsonl"
    RUN._write_jsonl_atomic(jsonl, rows)
    states = RUN.capture_answer_states(cfg, model, tok, flat_ctx, flat_text, eot)
    for r, n_tok in zip(rows, states["n_completion_tokens"], strict=True):
        r["n_completion_tokens"] = n_tok
        r["cap_hit"] = RUN.cap_hit(n_tok, cfg.max_new_tokens)
        r["cap_hit_basis"] = "retokenized_completion_len >= max_new_tokens"
    RUN._write_jsonl_atomic(jsonl, rows)
    RUN._save_pt_atomic(
        cfg.anchors_dir / f"va_anchors_gate_w{cfg.worker_index}.pt",
        {
            "layers": cfg.layers,
            "index": [{"context_id": r["context_id"], "draw": r["draw"]} for r in rows],
            "va_span": states["va_span"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "repro": RUN._repro(cfg),
        },
    )
    cap_hits = sum(1 for r in rows if r["cap_hit"])
    RUN._write_json_atomic(
        cfg.manifest_dir / f"anchors_gate_w{cfg.worker_index}_done.json",
        {
            "regime_fp": regime_fp,
            "batch": "gate",
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,
            "n_contexts": len(my_order),
            "draws": draws,
            "n_rows": len(rows),
            "n_cap_hit": cap_hits,
            "n_empty": len(states["empty_rows"]),
            "repro": RUN._repro(cfg),
        },
    )
    logger.info(
        "[anchors] rows=%d cap_hit=%d empty=%d", len(rows), cap_hits, len(states["empty_rows"])
    )
    # Immediate upload: the VM gate wave starts on these texts (plan §9 DAG).
    RUN._upload_dir(cfg, cfg.anchors_dir, f"{HF_LADDER_RAW}/anchors", [jsonl.name])
    logger.info("[phase=anchors_done] worker=%d", cfg.worker_index)
    return RUN.RC_OK


# ── P3: the grid (parent claim-file queue) ────────────────────────────


def _load_ladder_bank_states(cfg: LadderConfig) -> dict:
    path = cfg.bank_dir / "vc_bank_ladder.pt"
    assert path.exists(), f"{path} missing — run `--phase bank` first"
    return torch.load(path, map_location="cpu", weights_only=False)


def _grid_inputs(
    cfg: LadderConfig,
) -> tuple[
    dict, dict, list[LB.LadderPair], dict[str, dict[str, str]], list[str], list[RUN.Block], str
]:
    """Shared grid/margin setup: manifest, banks, pairs, donor maps, blocks."""
    manifest, bank_sha = _load_ladder_manifest(cfg)
    regime_fp = RUN.regime_fingerprint(cfg, bank_sha)
    pairs = pairs_from_manifest(manifest)
    if cfg.smoke:
        survivors = None
        screen = None
        blocks = smoke_ladder_blocks(pairs)
    else:
        assert cfg.gate_verdict_path is not None and cfg.gate_verdict_path.exists(), (
            f"--gate-verdict required for a non-smoke grid (got {cfg.gate_verdict_path}) — "
            "the anchor-separation gate runs BEFORE any grid spend (plan §4.5)"
        )
        assert cfg.donor_screen_path is not None and cfg.donor_screen_path.exists(), (
            f"--donor-screen required for a non-smoke grid (got {cfg.donor_screen_path}) — "
            "cross-type donors are construct-screened before the grid (plan §4.2)"
        )
        survivors = read_gate_verdict(cfg.gate_verdict_path)
        screen = read_donor_screen(cfg.donor_screen_path)
        blocks = None  # built after donor maps (needs the dropped set)
    donor_maps, dropped = donor_maps_ladder(manifest, pairs, survivors, screen)
    if blocks is None:
        blocks = enumerate_ladder_blocks(pairs, survivors, set(dropped))
    return manifest, {"survivors": survivors}, pairs, donor_maps, dropped, blocks, regime_fp


def _block_cells_ladder(
    bank: dict,
    block: RUN.Block,
    pairs_by_id: dict[str, LB.LadderPair],
    donor_maps: dict[str, dict[str, str]],
    parent_recs: dict,
) -> list[dict]:
    """Per-pair cell specs (payload, position, provenance) for one block."""
    recs = bank["per_context"]
    cells: list[dict] = []
    for pid in block.pair_ids:
        pair = pairs_by_id[pid]
        payload, donor_id = payload_for_arm_ladder(
            bank, pair, block.slot, block.arm, donor_maps, pairs_by_id, parent_recs=parent_recs
        )
        rec = recs[pair.a]
        cells.append(
            {
                "pair_id": pid,
                "pair": pair,
                "context_a": pair.a,
                "context_b": pair.b,
                "position": RUN.slot_position(rec["ctx_len"], rec["prefix_end"], block.slot),
                "payload": payload,
                "donor_context_id": donor_id,
                # Plan §4.1 length covariate: per-rung system-prompt length varies.
                "len_delta": int(recs[pair.b]["ctx_len"]) - int(rec["ctx_len"]),
            }
        )
    return cells


def _block_margin_rows_ladder(
    cfg: LadderConfig,
    model,
    tok,
    block: RUN.Block,
    cells: list[dict],
    pools: dict[str, list[dict]],
) -> list[dict]:
    """Grid margin TF under the PATCHED state (pool key = direction id)."""
    rows_spec: list[dict] = []
    meta: list[dict] = []
    out: list[dict] = []
    items = pools.get(block.cell)
    for cell in cells:
        pair: LB.LadderPair = cell["pair"]
        if not items:
            out.append(
                {
                    "block_key": block.key,
                    "pair_id": pair.pair_id,
                    "arm": block.arm,
                    "pool_key": block.cell,
                    "skipped": True,
                    "reason": "no pool for this direction (judge-filter yield below floor)",
                }
            )
            continue
        ctx_ids = _ctx_ids_for_pair(cfg, tok, cell)
        for idx, it in enumerate(items):
            item_ids = tok(it["text"], add_special_tokens=False)["input_ids"]
            assert item_ids, (block.cell, idx, "pool item tokenized empty")
            rows_spec.append(
                {
                    "ctx_ids": ctx_ids,
                    "item_ids": item_ids,
                    "payload": cell["payload"],
                    "position": cell["position"],
                }
            )
            meta.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "slot": block.slot,
                    "arm": block.arm,
                    "pair_id": pair.pair_id,
                    "donor_context_id": cell["donor_context_id"],
                    "pool_key": block.cell,
                    "pool_idx": idx,
                    "pool_side": it["side"],
                    "n_pool_tokens": len(item_ids),
                }
            )
    if rows_spec:
        lnps = RUN.margin_lnp(cfg, model, tok, rows_spec)
        for m, lnp in zip(meta, lnps, strict=True):
            out.append({**m, "lnp_mean": lnp, "skipped": False})
    return out


_CTX_IDS_CACHE: dict[str, list[int]] = {}


def _ctx_ids_for_pair(cfg: LadderConfig, tok, cell: dict) -> list[int]:
    cid = cell["context_a"]
    if cid not in _CTX_IDS_CACHE:
        _CTX_IDS_CACHE[cid] = BANK94.context_token_ids_2094(tok, cell["_contexts"][cid])
    return _CTX_IDS_CACHE[cid]


@torch.no_grad()
def run_ladder_block(
    cfg: LadderConfig,
    model,
    tok,
    bank: dict,
    block: RUN.Block,
    pairs_by_id: dict[str, LB.LadderPair],
    donor_maps: dict[str, dict[str, str]],
    parent_recs: dict,
    contexts: dict[str, dict],
    ctx_ids_cache: dict[str, list[int]],
    eot: list[int],
    regime_fp: str,
    pools: dict[str, list[dict]] | None,
    draws: int,
) -> dict:
    """One block: K hooked temp-1.0 draws per pair + the hooked V_a pass +
    (pools present) the inline margin TF pass (the parent ``run_block`` shape)."""
    cells = _block_cells_ladder(bank, block, pairs_by_id, donor_maps, parent_recs)
    for c in cells:
        c["_contexts"] = contexts  # margin ctx-id resolution seam

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK94.context_token_ids_2094(tok, contexts[cid])
        return ctx_ids_cache[cid]

    texts_per_cell: list[list[str]] = []
    for start in range(0, len(cells), cfg.gen_batch):
        chunk = cells[start : start + cfg.gen_batch]
        ctx_list = [contexts[c["context_a"]] for c in chunk]
        rows = [ids_for(c["context_a"]) for c in chunk]
        row_lengths = [len(r) for r in rows]
        t_pad = max(row_lengths)
        stack = RUN._arm_hook_all_layers(
            model,
            cfg,
            row_lengths,
            [(c["position"],) for c in chunk],
            [c["payload"] for c in chunk],
            t_pad,
        )
        try:
            outs = generate_batch(
                model,
                tok,
                ctx_list,
                n=draws,
                hook=stack,
                max_new_tokens=cfg.max_new_tokens,
                temperature=RUN.GRID_TEMPERATURE,
                seed_base=cfg.seed_base,
                render_fn=BANK94.render_context_2094,
                ids_fn=BANK94.context_token_ids_2094,
            )
        finally:
            stack.remove()
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        texts_per_cell.extend(list(o) for o in outs)
    assert len(texts_per_cell) == len(cells)

    flat_ctx: list[list[int]] = []
    flat_text: list[str] = []
    flat_payload: list[torch.Tensor] = []
    flat_pos: list[int] = []
    for c, texts in zip(cells, texts_per_cell, strict=True):
        for text in texts:
            flat_ctx.append(ids_for(c["context_a"]))
            flat_text.append(text)
            flat_payload.append(c["payload"])
            flat_pos.append(c["position"])
    states = RUN.capture_answer_states(
        cfg, model, tok, flat_ctx, flat_text, eot, payloads=flat_payload, positions=flat_pos
    )

    rows_out: list[dict] = []
    k = 0
    for c, texts in zip(cells, texts_per_cell, strict=True):
        pair: LB.LadderPair = c["pair"]
        for i, text in enumerate(texts):
            n_tok = states["n_completion_tokens"][k]
            rows_out.append(
                {
                    "block_key": block.key,
                    "cell": block.cell,
                    "direction": block.cell,
                    "kind": pair.kind,
                    "persona": pair.persona,
                    "slot": block.slot,
                    "arm": block.arm,
                    "pair_id": pair.pair_id,
                    "carrier": pair.carrier,
                    "value_a": pair.value_a,
                    "value_b": pair.value_b,
                    "context_a": pair.a,
                    "context_id": pair.a,
                    "context_b": pair.b,
                    "position": c["position"],
                    "donor_context_id": c["donor_context_id"],
                    "len_delta": c["len_delta"],
                    "draw": i,
                    "seed": cfg.seed_base + i,
                    "temperature": RUN.GRID_TEMPERATURE,
                    "n_completion_tokens": n_tok,
                    "cap_hit": RUN.cap_hit(n_tok, cfg.max_new_tokens),
                    "cap_hit_basis": "retokenized_completion_len >= max_new_tokens",
                    "text": text,
                }
            )
            k += 1
    RUN._write_jsonl_atomic(cfg.rollouts_dir / f"shard_{block.slug}.jsonl", rows_out)
    RUN._save_pt_atomic(
        cfg.va_dir / f"shard_{block.slug}.pt",
        {
            "block_key": block.key,
            "layers": cfg.layers,
            "index": [
                {"pair_id": r["pair_id"], "context_a": r["context_a"], "draw": r["draw"]}
                for r in rows_out
            ],
            "va_span": states["va_span"],
            "pooling": states["pooling"],
            "empty_rows": states["empty_rows"],
            "hooked_capture": True,
            "repro": RUN._repro(cfg),
        },
    )
    margin_done = False
    if pools is not None:
        margin_rows = _block_margin_rows_ladder(cfg, model, tok, block, cells, pools)
        RUN._write_jsonl_atomic(cfg.margin_dir / f"shard_{block.slug}.jsonl", margin_rows)
        RUN._write_json_atomic(
            RUN.block_done_path(cfg.out_root, block, "margin_blocks"),
            {
                "key": block.key,
                "regime_fp": regime_fp,
                "n_rows": len(margin_rows),
                "n_skipped": sum(1 for r in margin_rows if r.get("skipped")),
                "repro": RUN._repro(cfg),
            },
        )
        margin_done = True
    done = {
        "key": block.key,
        "regime_fp": regime_fp,
        "n_cells": block.n_pairs,
        "n_rows": len(rows_out),
        "n_cap_hit": sum(1 for r in rows_out if r["cap_hit"]),
        "n_empty": len(states["empty_rows"]),
        "margin_inline": margin_done,
        "repro": RUN._repro(cfg),
    }
    RUN._write_json_atomic(RUN.block_done_path(cfg.out_root, block), done)
    return done


def phase_grid(cfg: LadderConfig) -> int:
    """P3: claim-queue block execution over the gate survivors."""
    logger.info("[phase=grid] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke)
    manifest, meta, pairs, donor_maps, dropped, blocks, regime_fp = _grid_inputs(cfg)
    bank = _load_ladder_bank_states(cfg)
    _parent_manifest, parent_recs = load_parent_bank(cfg)
    pairs_by_id = {p.pair_id: p for p in pairs}
    contexts = manifest["contexts"]
    draws = SMOKE_GRID_DRAWS if cfg.smoke else cfg.grid_draws
    pools: dict[str, list[dict]] | None = None
    if cfg.pools_path is not None and cfg.pools_path.exists():
        pools = RUN.load_pools(cfg.pools_path)
        logger.info("[grid] margin pools loaded: %d pools (%s)", len(pools), cfg.pools_path)
    else:
        logger.info(
            "[grid] no pools file (%s) — margins deferred to --phase margin", cfg.pools_path
        )
    totals = RUN.grid_totals(blocks, draws)
    RUN._write_json_atomic(
        cfg.manifest_dir / f"grid_plan_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "num_workers": cfg.num_workers,
            "smoke": cfg.smoke,
            "survivors": meta["survivors"],
            "dropped_xtype_pairs": sorted(dropped),
            "totals_this_run": totals,
            "queue": "shared claim-file queue (work-conserving; parent skeleton)",
            "margin_inline": pools is not None,
            "repro": RUN._repro(cfg),
        },
    )
    logger.info(
        "[grid] %d blocks / %d cells / %d rollouts (dropped xtype pairs: %d)",
        totals["n_blocks"],
        totals["cells_total"],
        totals["rollouts_total"],
        len(dropped),
    )
    model, tok = RUN.load_model_and_tokenizer(cfg)
    eot = RUN.eot_tail_ids(tok)
    ctx_ids_cache: dict[str, list[int]] = {}
    ran_rollouts = 0
    n_run = 0
    uploaded: list[str] = []
    pending: list[RUN.Block] = []

    def run_one(block: RUN.Block) -> None:
        nonlocal ran_rollouts, n_run, uploaded, pending
        t0 = time.monotonic()
        rec = run_ladder_block(
            cfg,
            model,
            tok,
            bank,
            block,
            pairs_by_id,
            donor_maps,
            parent_recs,
            contexts,
            ctx_ids_cache,
            eot,
            regime_fp,
            pools,
            draws,
        )
        ran_rollouts += rec["n_rows"]
        n_run += 1
        pending.append(block)
        logger.info(
            "[grid] unit %d %s rows=%d cap_hit=%d elapsed=%.1fs",
            n_run,
            block.key,
            rec["n_rows"],
            rec["n_cap_hit"],
            time.monotonic() - t0,
        )
        if cfg.upload_every > 0 and len(pending) >= cfg.upload_every:
            uploaded += _upload_grid_increment_ladder(cfg, pending)
            pending.clear()

    stats = RUN.run_claim_queue(cfg, blocks, regime_fp, "blocks", run_one)
    if pending:
        uploaded += _upload_grid_increment_ladder(cfg, pending)
        pending.clear()
    RUN._write_json_atomic(
        cfg.manifest_dir / f"grid_done_w{cfg.worker_index}.json",
        {
            "regime_fp": regime_fp,
            "worker_index": cfg.worker_index,
            "n_blocks_run": stats["ran"],
            "n_rollouts_run": ran_rollouts,
            "queue_waits": stats["waits"],
            "uploads": uploaded,
            "repro": RUN._repro(cfg),
        },
    )
    logger.info(
        "[phase=grid_done] worker=%d blocks_run=%d rollouts=%d",
        cfg.worker_index,
        stats["ran"],
        ran_rollouts,
    )
    return RUN.RC_OK


def _upload_grid_increment_ladder(cfg: LadderConfig, blocks: list[RUN.Block]) -> list[str]:
    slugs = [b.slug for b in blocks if (cfg.rollouts_dir / f"shard_{b.slug}.jsonl").exists()]
    if not slugs:
        return []
    return RUN._upload_dir(
        cfg, cfg.rollouts_dir, f"{HF_LADDER_RAW}/grid", [f"shard_{s}.jsonl" for s in slugs]
    )


# ── margin phase (pools-dependent TF legs) ────────────────────────────


def phase_margin(cfg: LadderConfig) -> int:
    """Margin TF: (a) anchor margins (every context x its directions' pools,
    unhooked) and (b) the per-block catch-up (claim namespace ``margin_blocks``)."""
    logger.info(
        "[phase=margin] worker=%d/%d smoke=%s", cfg.worker_index, cfg.num_workers, cfg.smoke
    )
    assert cfg.pools_path is not None and cfg.pools_path.exists(), (
        f"--pools file required for --phase margin (got {cfg.pools_path}) — pools are "
        "judge-built from the gate wave and staged by the orchestrator"
    )
    pools = RUN.load_pools(cfg.pools_path)
    manifest, meta, pairs, donor_maps, dropped, blocks, regime_fp = _grid_inputs(cfg)
    bank = _load_ladder_bank_states(cfg)
    _parent_manifest, parent_recs = load_parent_bank(cfg)
    pairs_by_id = {p.pair_id: p for p in pairs}
    contexts = manifest["contexts"]
    model, tok = RUN.load_model_and_tokenizer(cfg)
    ctx_ids_cache: dict[str, list[int]] = {}

    def ids_for(cid: str) -> list[int]:
        if cid not in ctx_ids_cache:
            ctx_ids_cache[cid] = BANK94.context_token_ids_2094(tok, contexts[cid])
        return ctx_ids_cache[cid]

    # (a) anchor margins — unhooked TF of each context's directions' pools.
    done_key = f"margin_anchors_w{cfg.worker_index}"
    if cfg.force or RUN._sharded_done_record(cfg, done_key, regime_fp) is None:
        pool_keys_by_ctx: dict[str, list[str]] = {}
        for p in pairs:
            for cid in (p.a, p.b):
                keys = pool_keys_by_ctx.setdefault(cid, [])
                if p.cell not in keys:
                    keys.append(p.cell)
        order = [cid for cid in contexts if cid in pool_keys_by_ctx]
        my_order = order[cfg.worker_index :: cfg.num_workers]
        rows_spec: list[dict] = []
        meta_rows: list[dict] = []
        out_rows: list[dict] = []
        for cid in my_order:
            for key in pool_keys_by_ctx[cid]:
                items = pools.get(key)
                if not items:
                    out_rows.append(
                        {
                            "context_id": cid,
                            "pool_key": key,
                            "skipped": True,
                            "reason": "no pool for this direction",
                        }
                    )
                    continue
                for idx, it in enumerate(items):
                    item_ids = tok(it["text"], add_special_tokens=False)["input_ids"]
                    assert item_ids, (key, idx, "pool item tokenized empty")
                    rows_spec.append(
                        {
                            "ctx_ids": ids_for(cid),
                            "item_ids": item_ids,
                            "payload": None,
                            "position": None,
                        }
                    )
                    meta_rows.append(
                        {
                            "context_id": cid,
                            "value_id": contexts[cid]["value_id"],
                            "rung": contexts[cid]["rung"],
                            "carrier": contexts[cid]["carrier"],
                            "pool_key": key,
                            "pool_idx": idx,
                            "pool_side": it["side"],
                            "n_pool_tokens": len(item_ids),
                        }
                    )
        if rows_spec:
            t0 = time.monotonic()
            lnps = RUN.margin_lnp(cfg, model, tok, rows_spec)
            logger.info("[margin:anchors] %d rows in %.1fs", len(rows_spec), time.monotonic() - t0)
            out_rows.extend(
                {**m, "lnp_mean": lnp, "skipped": False}
                for m, lnp in zip(meta_rows, lnps, strict=True)
            )
        RUN._write_jsonl_atomic(
            cfg.margin_dir / f"anchor_margin_w{cfg.worker_index}.jsonl", out_rows
        )
        RUN._write_json_atomic(
            cfg.manifest_dir / f"{done_key}_done.json",
            {
                "regime_fp": regime_fp,
                "worker_index": cfg.worker_index,
                "num_workers": cfg.num_workers,
                "n_rows": len(out_rows),
                "repro": RUN._repro(cfg),
            },
        )
    else:
        logger.info("[margin:anchors] already done for this regime — skipping")

    # (b) per-block catch-up via the parent claim queue (namespace matches the
    # inline-margin done files, so grid-inline blocks are skipped).
    def run_one(block: RUN.Block) -> None:
        cells = _block_cells_ladder(bank, block, pairs_by_id, donor_maps, parent_recs)
        for c in cells:
            c["_contexts"] = contexts
            ids_for(c["context_a"])
        t0 = time.monotonic()
        margin_rows = _block_margin_rows_ladder(cfg, model, tok, block, cells, pools)
        RUN._write_jsonl_atomic(cfg.margin_dir / f"shard_{block.slug}.jsonl", margin_rows)
        RUN._write_json_atomic(
            RUN.block_done_path(cfg.out_root, block, "margin_blocks"),
            {
                "key": block.key,
                "regime_fp": regime_fp,
                "n_rows": len(margin_rows),
                "n_skipped": sum(1 for r in margin_rows if r.get("skipped")),
                "repro": RUN._repro(cfg),
            },
        )
        logger.info(
            "[margin] unit %s rows=%d elapsed=%.1fs",
            block.key,
            len(margin_rows),
            time.monotonic() - t0,
        )

    stats = RUN.run_claim_queue(cfg, blocks, regime_fp, "margin_blocks", run_one)
    logger.info("[phase=margin_done] worker=%d blocks_run=%d", cfg.worker_index, stats["ran"])
    return RUN.RC_OK


# ── P4: upload + sentinel ─────────────────────────────────────────────


def _margin_state_ladder(cfg: LadderConfig) -> dict:
    """Disk-derived margin completeness (the parent ``_margin_state`` shape).

    The ladder's EXPECTED block set depends on the anchor-separation gate
    verdict (only survivor blocks run — plan §4.3), which the upload phase
    does not carry, so the expected set is derived from DISK: every grid
    block that COMPLETED (a ``blocks/`` done-file) must have its
    ``margin_blocks/`` twin. A fresh deferred-leg pod (no local grid state)
    reads complete-on-blocks and is stamped via ``deferred_leg``."""
    grid_done = sorted((cfg.manifest_dir / "blocks").glob("*.done.json"))
    blocks_done = sum(
        1 for p in grid_done if (cfg.manifest_dir / "margin_blocks" / p.name).exists()
    )
    recs: list[dict] = []
    for p in sorted(cfg.manifest_dir.glob("margin_anchors_w*_done.json")):
        try:
            recs.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    anchors_done = False
    for w in {int(r.get("num_workers", 0)) for r in recs}:
        idxs = {int(r.get("worker_index", -1)) for r in recs if int(r.get("num_workers", 0)) == w}
        if w > 0 and idxs >= set(range(w)):
            anchors_done = True
    deferred = blocks_done < len(grid_done) or not anchors_done
    state: dict = {
        "margin_deferred": deferred,
        "margin_blocks_done": blocks_done,
        "margin_blocks_expected": len(grid_done),
        "margin_anchors_done": anchors_done,
    }
    if deferred:
        state["margin_deferred_recipe"] = (
            "build pools VM-side (uv run python scripts/issue2162_ladder_judge.py --phase pools), "
            "scp pools_ladder.json to the pod, then scripts/issue2162_ladder.py --phase margin "
            "--pools <path> --gate-verdict <path> --donor-screen <path> && --phase upload"
        )
    return state


def _sentinel_payload_ladder(cfg: LadderConfig, uploaded: dict[str, list[str]]) -> dict:
    """The /issue Step 7 results payload (parent 10-key shape, ladder paths)."""
    n_grid_shards = len(list(cfg.rollouts_dir.glob("shard_*.jsonl")))
    n_va_shards = len(list(cfg.va_dir.glob("shard_*.pt")))
    n_margin_shards = len(list(cfg.margin_dir.glob("*.jsonl")))
    n_anchor_rows = 0
    for jsonl in sorted(cfg.anchors_dir.glob("anchors_*.jsonl")):
        n_anchor_rows += sum(1 for line in jsonl.open(encoding="utf-8") if line.strip())
    gate_path = cfg.gates_dir / "injection_gate_report.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    distinct_path = cfg.gates_dir / "distinctness_report.json"
    distinct = json.loads(distinct_path.read_text()) if distinct_path.exists() else {}
    block_done_recs = sorted((cfg.manifest_dir / "blocks").glob("*.done.json"))
    cap_hits, rows_total = 0, 0
    for done in block_done_recs:
        rec = json.loads(done.read_text())
        cap_hits += int(rec.get("n_cap_hit", 0))
        rows_total += int(rec.get("n_rows", 0))
    margin_state = _margin_state_ladder(cfg)
    return {
        **margin_state,
        "deferred_leg": not block_done_recs,
        "eval_numbers": {
            "round": "persona-specificity-ladder",
            "grid_shards": n_grid_shards,
            "va_shards": n_va_shards,
            "margin_shards": n_margin_shards,
            "anchor_rows": n_anchor_rows,
            "grid_rollouts_persisted": rows_total,
            "cap_hit_rows": cap_hits,
            "cap_hit_frac": (cap_hits / rows_total) if rows_total else 0.0,
            "injection_gate_passed": bool(gate.get("passed")),
            "injection_gate_spots_failed": int(gate.get("n_spots_failed", 0)),
            "distinctness_gate_passed": bool(distinct.get("passed")),
            "distinctness_violations": int(distinct.get("n_violations", 0)),
        },
        "eval_paths": sorted(
            {
                str(cfg.bank_dir / "ladder_bank.json"),
                str(cfg.bank_dir / "vc_bank_ladder.pt"),
                str(gate_path),
                str(distinct_path),
                str(cfg.anchors_dir),
                str(cfg.rollouts_dir),
                str(cfg.va_dir),
                str(cfg.margin_dir),
            }
        ),
        "reproducibility_card": {
            **RUN._repro(cfg),
            "seed_base": cfg.seed_base,
            "bank_seed": LB.SEED,
            "parent_hf_revision": cfg.hf_revision,
            "max_new_tokens": cfg.max_new_tokens,
            "grid_temperature": RUN.GRID_TEMPERATURE,
            "grid_draws": cfg.grid_draws,
            "anchor_temperature": RUN.ANCHOR_TEMPERATURE,
            "anchor_draws": cfg.anchor_draws,
            "gen_batch": cfg.gen_batch,
            "num_workers": cfg.num_workers,
        },
        "wandb_url": None,
        "hf_hub_url": (
            f"https://huggingface.co/datasets/{RUN.HF_DATA_REPO}/tree/main/{HF_LADDER_TENSORS}"
        ),
        "worktree_path": str(RUN.REPO_ROOT),
        "final_commit_sha": RUN._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": cfg.gpu_hours_budgeted,
        "plan_deviations": [
            "margin shards persist as JSONL (scalar lnP rows — the parent margin "
            "format), uploaded under analysis_tensors/ladder/margin/ (plan §6.5 "
            "wrote '*.pt' for this slot)",
            "cap-hit telemetry derived from the re-tokenized completion length "
            "(generate_batch returns decoded text only) — parent convention",
        ],
        "uploaded_prefixes": {k: len(v) for k, v in uploaded.items()},
    }


def phase_upload(cfg: LadderConfig) -> int:
    """P4: bulk upload every ladder prefix, then write the pod sentinel."""
    logger.info("[phase=upload]")
    uploaded: dict[str, list[str]] = {}
    uploaded["vc_bank"] = RUN._upload_dir(
        cfg, cfg.bank_dir, f"{HF_LADDER_TENSORS}/vc_bank", ["*.pt", "*.json"]
    )
    uploaded["gates"] = RUN._upload_dir(
        cfg, cfg.gates_dir, f"{HF_LADDER_TENSORS}/gates", ["*.json"]
    )
    uploaded["anchors_text"] = RUN._upload_dir(
        cfg, cfg.anchors_dir, f"{HF_LADDER_RAW}/anchors", ["*.jsonl"]
    )
    uploaded["anchors_tensors"] = RUN._upload_dir(
        cfg, cfg.anchors_dir, f"{HF_LADDER_TENSORS}/anchors", ["*.pt"]
    )
    uploaded["grid_text"] = RUN._upload_dir(
        cfg, cfg.rollouts_dir, f"{HF_LADDER_RAW}/grid", ["shard_*.jsonl"]
    )
    uploaded["va_store"] = RUN._upload_dir(
        cfg, cfg.va_dir, f"{HF_LADDER_TENSORS}/va_store", ["shard_*.pt"]
    )
    uploaded["margin"] = RUN._upload_dir(
        cfg, cfg.margin_dir, f"{HF_LADDER_TENSORS}/margin", ["*.jsonl"]
    )
    uploaded["manifests"] = RUN._upload_dir(
        cfg,
        cfg.manifest_dir,
        f"{HF_LADDER_TENSORS}/manifests",
        ["*.json", "blocks/*.done.json", "margin_blocks/*.done.json"],
    )
    payload = _sentinel_payload_ladder(cfg, uploaded)
    if payload["margin_deferred"]:
        logger.warning(
            "[upload] margin DEFERRED (blocks %d/%d, anchors_done=%s) — sentinel records "
            "margin_deferred=true + the recipe; teardown proceeds",
            payload["margin_blocks_done"],
            payload["margin_blocks_expected"],
            payload["margin_anchors_done"],
        )
    RUN._write_json_atomic(cfg.out_root / "manifests" / "upload_done.json", payload)
    sentinel = cfg.log_dir / (RUN.SENTINEL_NAME_SMOKE if cfg.smoke else RUN.SENTINEL_NAME)
    body = {
        "sentinel_schema_version": 1,
        "kind": "epm:smoke-result" if cfg.smoke else "epm:results",
        "version": 1,
        "note": payload,
    }
    RUN._write_json_atomic(sentinel, body)
    logger.info("[upload] sentinel written: %s", sentinel)
    logger.info("[phase=upload_done]")
    return RUN.RC_OK


# ── entrypoint ────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve EVERY deferred import this driver reaches on its real paths
    (staging: huggingface_hub + retry_transient; model: transformers; upload:
    hub helpers via RUN), plus the ladder registry invariants."""
    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    import transformers  # noqa: F401
    from explore_persona_space.orchestrate.hub import (  # noqa: F401
        _upload_folder_filtered,
        retry_transient,
        verify_repo_paths_uploaded,
    )

    assert len(LB.LADDER_VALUES) == 7
    assert len(LB.carrier_ids()) == 6
    assert len(LB.direction_ids()) == 12
    assert len(LB.build_ladder_pairs()) == 72
    assert set(LB.rubric_registry()) == {
        LB.holistic_rubric_id(v.value_id) for v in LB.LADDER_VALUES
    }
    # Argparse-attribute completeness (orchestrate.argcheck; code-style.md).
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("[import-check] OK", flush=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return RUN.RC_OK
    assert args.phase, "--phase is required (or pass --import-check)"
    cfg = build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if args.gpu_id is not None:
        logger.info(
            "[env] --gpu-id=%s CUDA_VISIBLE_DEVICES=%s",
            args.gpu_id,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    if cfg.phase == "bank":
        return phase_bank(cfg)
    if cfg.phase == "anchors":
        return phase_anchors(cfg)
    if cfg.phase == "grid":
        return phase_grid(cfg)
    if cfg.phase == "margin":
        return phase_margin(cfg)
    assert cfg.phase == "upload", cfg.phase
    return phase_upload(cfg)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
