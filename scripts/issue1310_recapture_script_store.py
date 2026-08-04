"""Issue #1310 script-format store RE-CAPTURE (store gap repair, 2026-07-30).

The run-2 SCRIPT-format activation store — the substrate of the published
per-persona cells `eval_results/issue_1310/cells_{base,instruct}_<persona>.json`
(ambient pure-GCV, no `gcv_dof_cap` field) and
`script_completion/cells_scriptc_instruct_Vex.json` — was lost with its
instance, so the `nd-estimator-audit` round could not recompute those cells.
This driver rebuilds it from the PERSISTED story text and re-uploads it under a
NEW name (`store_recap`), overwriting nothing.

Phases (blocking, in order; every phase prints its own breadcrumb):

  p0_stage        stage story text (both models) + the persisted script-format
                  instruct turn-pairs from the HF data repo
  p1_pairs        re-attribute the BASE turn-pairs deterministically (the base
                  pairs were never persisted) and gate BOTH arms' per-persona
                  counts against the published cell n
  p2_capture      `issue1310_extract_store.py --flavor perturn` per model into
                  `<data-dir>/store_recap/<model>` (28 layers, bf16, the same
                  rig the lost store was built with)
  p3_upload       one `upload_folder` commit per model + an EXACT-SET verify,
                  BEFORE any instance teardown (the loss class being repaired)

Allocation-safe: narrows to the FIRST device of the pre-set allocation and never
exports an absolute GPU index (the `issue1345_boundary_ablation_launch_gen.sh`
pattern). Single-GPU by construction — one model loaded at a time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + tokens before any heavy import (#847)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_ROOT = "issue1310_char_map"
STORE_SUBDIR = "store_recap"
STORE_PREFIX = f"{HF_ROOT}/analysis_tensors/{STORE_SUBDIR}"
DATA_DIR = REPO / "data" / "issue_1310"
SENTINEL = Path("/workspace/logs/issue-1310-results.json")

# Published per-persona turn-pair counts (the audited cells' own n).
# base: eval_results/issue_1310/cells_base_<persona>.json
# instruct: cells_instruct_{Wren,HELIOS,Dana}.json +
#           script_completion/cells_scriptc_instruct_Vex.json
PUBLISHED_PAIRS = {
    "base": {"Wren": 2329, "HELIOS": 2466, "Dana": 1325, "Vex": 2060},
    "instruct": {"Wren": 3094, "HELIOS": 3123, "Dana": 2700, "Vex": 3586},
}
COUNT_TOLERANCE = 0.05  # report + fail loud above 5% divergence (round brief)


def _sh(cmd: list[str], *, phase: str) -> None:
    """Run a blocking subprocess, fail loud with the phase name."""
    print(f"[recap] {phase}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    print(f"[recap] {phase}: rc={rc} elapsed={time.time() - t0:.0f}s", flush=True)
    if rc != 0:
        raise RuntimeError(f"{phase} failed rc={rc}")


def _first_allocated_device() -> str | None:
    """First device of the pre-set allocation; None when unset (whole node).

    NEVER exports an absolute index: an allocation-scoped launcher may hand us
    e.g. CUDA_VISIBLE_DEVICES="3,5", where torch's cuda:0 is physical 3.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return None
    return cvd.split(",")[0].strip() or None


def phase_stage() -> None:
    """Stage story text (both models) + the persisted instruct pairs from HF."""
    print("[phase=p0_stage] staging persisted script-format inputs", flush=True)
    from explore_persona_space.orchestrate import hub

    (DATA_DIR / "stories").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "pairs").mkdir(parents=True, exist_ok=True)
    spec = {
        DATA_DIR
        / "stories"
        / "base_stories_seed42.jsonl": f"{HF_ROOT}/raw_completions/generation/base_stories_seed42.jsonl",
        DATA_DIR
        / "stories"
        / "instruct_stories_seed42.jsonl": f"{HF_ROOT}/raw_completions/generation/instruct_stories_seed42.jsonl",
        # NOTE: the completion round's persisted instruct pairs
        # (raw_completions/pairs_script_completion/instruct_pairs.jsonl) are
        # DELIBERATELY NOT staged. They are mutually INCOHERENT with the
        # persisted instruct story text: 1,241/12,503 (9.93%) of their token
        # spans overflow the token length of the story row they name, across
        # 457/1,180 scenes, with the worst reaching hi=1,024 (the generation
        # cap) against a 557-token story — i.e. they were attributed against a
        # LONGER instruct story set than the one persisted under
        # `generation/`, which is not recoverable. Feeding them to the capture
        # rig raises `PairSpec.validate` (observed: job 16081,
        # `AssertionError: ('sc_0005:Wren:t013', 't_span', 603, 630, 616)`).
        # Both arms are therefore re-attributed from the persisted stories, so
        # pairs and text are coherent by construction (artifact-reuse check
        # (j), pairwise provenance coherence).
    }
    for target, path_in_repo in spec.items():
        hub.stage_hub_file(DATA_REPO, path_in_repo, target, repo_type="dataset")
        n = sum(1 for ln in target.open(encoding="utf-8") if ln.strip())
        print(f"[stage] {target.name} staged: {n} rows", flush=True)


def _pair_counts(model_kind: str) -> dict[str, int]:
    """Per-persona turn-pair counts from the staged pairs JSONL."""
    path = DATA_DIR / "pairs" / f"{model_kind}_pairs.jsonl"
    counts: dict[str, int] = {}
    # text-mode iteration, never .splitlines() (raw U+2028 in real text, #950)
    for line in path.open(encoding="utf-8"):
        if line.strip():
            char_id = json.loads(line)["char_id"]
            counts[char_id] = counts.get(char_id, 0) + 1
    return counts


def _gate_counts(model_kind: str) -> dict:
    """Compare realized per-persona pair counts against the published cell n."""
    realized = _pair_counts(model_kind)
    published = PUBLISHED_PAIRS[model_kind]
    rows, worst = {}, 0.0
    for persona, want in published.items():
        got = realized.get(persona, 0)
        rel = abs(got - want) / want
        worst = max(worst, rel)
        rows[persona] = {"published": want, "realized": got, "rel_divergence": rel}
        print(
            f"[gate] {model_kind}/{persona}: published={want} realized={got} rel={rel:.4f}",
            flush=True,
        )
    verdict = (
        "exact"
        if worst == 0.0
        else ("within-tolerance" if worst <= COUNT_TOLERANCE else "DIVERGENT")
    )
    print(f"[gate] {model_kind}: worst rel divergence {worst:.4f} -> {verdict}", flush=True)
    if worst > COUNT_TOLERANCE:
        raise RuntimeError(
            f"{model_kind} pair counts diverge from published by {worst:.4f} "
            f"> {COUNT_TOLERANCE} — refusing to capture; rows={rows}"
        )
    return {"per_persona": rows, "worst_rel_divergence": worst, "verdict": verdict}


def _gate_span_coherence(model_kind: str) -> dict:
    """Every pair's token spans must fit inside the story row it names.

    The check that would have caught job 16081's crash BEFORE the GPU spend:
    the completion round's persisted instruct pairs overflowed the persisted
    instruct story lengths on 9.93% of rows, and the capture rig only discovers
    that at `PairSpec.validate`, mid-forward-loop. Cheap (tokenize-only, CPU).
    """
    import issue1310_common as c1310

    tok = c1310.get_tokenizer(c1310.MODEL_IDS[model_kind])
    stories, pairs_by_scene = {}, {}
    story_path = DATA_DIR / "stories" / f"{model_kind}_stories_seed42.jsonl"
    for line in story_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            stories[row["row_id"]] = row["story"]
    for line in (DATA_DIR / "pairs" / f"{model_kind}_pairs.jsonl").open(encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            pairs_by_scene.setdefault(p["meta"]["scene_row_id"], []).append(p)
    n_over = n_tot = 0
    worst = None
    for scene, ps in pairs_by_scene.items():
        assert scene in stories, f"{model_kind}: pair scene {scene!r} absent from stories"
        n_story = len(tok(stories[scene], add_special_tokens=False)["input_ids"])
        for p in ps:
            n_tot += 1
            hi = max(max(h for _, h in p["t_spans"]), p["c_span"][1], p["ctx_span"][1])
            if hi > n_story:
                n_over += 1
                if worst is None or hi - n_story > worst[1]:
                    worst = (scene, hi - n_story, n_story, hi)
    frac = n_over / n_tot if n_tot else 0.0
    print(
        f"[gate] {model_kind} span-coherence: {n_over}/{n_tot} pairs overflow "
        f"({frac:.4%}); worst={worst}",
        flush=True,
    )
    if n_over:
        raise RuntimeError(
            f"{model_kind}: {n_over}/{n_tot} pair spans overflow their story's token "
            f"length (worst {worst}) — pairs and story text are INCOHERENT; refusing "
            "to capture"
        )
    return {"n_pairs": n_tot, "n_overflow": 0, "verdict": "coherent"}


def phase_pairs() -> dict:
    """Re-attribute BOTH arms from the persisted stories; gate counts + spans."""
    print("[phase=p1_pairs] re-attribution + count/span gates", flush=True)
    # Deterministic regex line attribution; the Sonnet leg is a PRECISION
    # spot-check only and plays no part in pair construction, so it is skipped
    # (the gates below are the binding checks). BOTH arms are re-attributed:
    # base pairs were never persisted, and the persisted instruct pairs are
    # span-incoherent with the persisted instruct stories (see phase_stage).
    for model_kind in ("base", "instruct"):
        _sh(
            [
                "uv",
                "run",
                "python",
                "scripts/issue1310_attribute.py",
                "--model",
                model_kind,
                "--data-dir",
                str(DATA_DIR),
                "--out-dir",
                str(REPO / "eval_results" / "issue_1310" / "recap"),
                "--skip-audit",
            ],
            phase=f"p1_pairs_{model_kind}_attribute",
        )
    return {
        k: {**_gate_counts(k), "span_coherence": _gate_span_coherence(k)}
        for k in ("base", "instruct")
    }


def _capture_one(model_kind: str) -> None:
    """Capture 28-layer span summaries for one model into the _recap store."""
    _sh(
        [
            "uv",
            "run",
            "python",
            "scripts/issue1310_extract_store.py",
            "--model",
            model_kind,
            "--data-dir",
            str(DATA_DIR),
            "--store-subdir",
            STORE_SUBDIR,
            "--flavor",
            "perturn",
            "--resume",
            "--equivalence-check",
        ],
        phase=f"p2_capture_{model_kind}",
    )


def _upload_one(model_kind: str) -> dict:
    """One upload_folder commit for this model's store dir + EXACT-SET verify."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    local = DATA_DIR / STORE_SUBDIR / model_kind
    files = sorted(p for p in local.rglob("*") if p.is_file())
    assert files, f"no store files under {local}"
    path_in_repo = f"{STORE_PREFIX}/{model_kind}"
    hub._upload(local, DATA_REPO, "dataset", path_in_repo, raise_on_error=True)
    expected = [f"{path_in_repo}/{p.relative_to(local).as_posix()}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), DATA_REPO, expected, path_in_repo=path_in_repo
    )
    total_gb = sum(p.stat().st_size for p in files) / 1e9
    print(
        f"[upload] {model_kind}: {len(files)} files ({total_gb:.2f} GB) -> "
        f"{path_in_repo}; missing={len(missing)}",
        flush=True,
    )
    if missing:
        raise RuntimeError(f"{model_kind} upload verify FAILED, missing: {missing[:8]}")
    return {
        "path_in_repo": path_in_repo,
        "n_files": len(files),
        "total_gb": round(total_gb, 3),
        "verify": "exact-set PASS",
    }


def phase_capture_and_upload(device: str | None) -> dict:
    """Capture then IMMEDIATELY upload+verify, per model.

    Per-model (not terminal-batch) upload is the #664 rule and the whole point
    of this round: a mid-run death after the base arm must not strand the base
    store the way the original run's whole store was stranded.
    """
    env_note = f"(device={device})" if device else "(whole-node allocation)"
    print(f"[phase=p2_capture_upload] capture + per-model durable upload {env_note}", flush=True)
    out: dict[str, dict] = {}
    for model_kind in ("base", "instruct"):
        _capture_one(model_kind)
        print(f"[phase=p3_upload_{model_kind}] store upload + exact-set verify", flush=True)
        out[model_kind] = _upload_one(model_kind)
    return out


def main() -> int:
    t0 = time.time()
    device = _first_allocated_device()
    if device is not None:
        # Narrow to ONE device of the pre-set allocation; never an absolute index.
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"[recap] narrowed CUDA_VISIBLE_DEVICES to allocated device {device}", flush=True)
    phase_stage()
    gates = phase_pairs()
    uploads = phase_capture_and_upload(device)
    payload = {
        "issue": 1310,
        "round": "nd-estimator-audit-recapture",
        "store_prefix": STORE_PREFIX,
        "pair_count_gates": gates,
        "uploads": uploads,
        "elapsed_s": round(time.time() - t0, 1),
    }
    try:
        SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        SENTINEL.write_text(json.dumps(payload, indent=1))
        print(f"[recap] wrote sentinel {SENTINEL}", flush=True)
    except OSError as exc:  # non-/workspace lane: the stdout payload is the record
        print(f"[recap] sentinel unwritable ({exc}); payload follows", flush=True)
    print("[recap] PAYLOAD " + json.dumps(payload), flush=True)
    print(f"[recap] complete in {payload['elapsed_s']}s", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize race (#1689)
