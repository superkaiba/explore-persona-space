#!/usr/bin/env python3
"""P1 vector builder for the #1902 steer_probe inline round (VM-side, CPU).

Recomputes the preimage-check geometry (scripts/issue1902_preimage_check.py,
same LadderContext loaders, full-sample standardized primal ridge B->S at
layer 31 / single / ctx) and SAVES the steering vectors the pod probe
(scripts/issue1902_steer_probe.py) consumes:

  c_star.npy   optimal constant correction mean(w_SS - f_BB(u_S))   (fp32, d)
  dy.npy       answer-cloud mean shift mean(w_SS) - mean(w_BB)      (fp32, d)
  v_pre.npy    minimal-norm strong-band (sigma >= sigma_max/10) context-side
               preimage of c_star through the fitted map, UNSTANDARDIZED
  v_rand.npy   fixed-seed random vector scaled to ||v_pre|| (null arm)

plus probe_inputs.jsonl (256 fixed-seed rows from the single-turn
intersection: context query + BASE greedy answer text from the parent gen
shards — data plumbing only, row text never printed) and meta.json
(norms / reachable fraction / predicted magnitude / provenance / sha256s).
Everything uploads to hf:{HF_DATA_REPO}/issue1902_stage_map/steer_probe/inputs/
in ONE upload_folder commit, exact-set verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1902_common as C  # noqa: E402
import issue1902_run as R  # noqa: E402
from issue1902_ladder_followup import ARM, CORPUS, LAYER_STAR, LadderContext  # noqa: E402

LADDER_OUT_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue1902_ladder")
STAGE_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue1902_steer_probe")
INPUTS_HF_DIR = f"{C.HF_PREFIX}/steer_probe/inputs"
STRONG_BAND_CUTOFF = 10.0  # sigma >= sigma_max/10 (preimage_check strong band)
N_PROBE_ROWS = 256
SEED = 1902

# Parity anchors from the committed diagnostic
# eval_results/issue_1902/followup_ladder/preimage_check.json (2026-08-04).
EXPECTED = {"c_star": 15.884, "dy": 18.046, "v_pre": 7.440, "frac": 0.760}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_and_vectors() -> tuple[dict[str, np.ndarray], dict]:
    """Full-sample standardized ridge (preimage_check recipe) -> vectors + stats."""
    lj = json.load(open(PROJECT_ROOT / "eval_results/issue_1902/followup_ladder/ladder_modes.json"))
    lam = float(np.median(lj["pairs"]["B->S"]["lambda_f_ii_per_fold"]))

    ctx = LadderContext(LADDER_OUT_ROOT, ["B", "S"], layer=LAYER_STAR)
    u_B, w_BB = ctx.xy("B", "B", CORPUS, LAYER_STAR, ARM)
    u_S, w_SS = ctx.xy("S", "S", CORPUS, LAYER_STAR, ARM)
    u_B, w_BB, u_S, w_SS = (np.asarray(a, dtype=np.float64) for a in (u_B, w_BB, u_S, w_SS))
    n, d = u_B.shape
    print(f"[vectors] n={n} d={d} lambda={lam:.4g}", flush=True)

    xmu, xsd = u_B.mean(0), u_B.std(0) + 1e-9
    Xn = (u_B - xmu) / xsd
    ymu = w_BB.mean(0)
    G = Xn.T @ Xn + lam * np.eye(d)
    W = np.linalg.solve(G, Xn.T @ (w_BB - ymu))  # (d, d): standardized ctx -> centered ans

    dy = w_SS.mean(0) - w_BB.mean(0)
    c_star = (w_SS - (((u_S - xmu) / xsd) @ W + ymu)).mean(0)

    # Preimage through M = W.T (ans <- std-ctx), strong band only.
    M = W.T
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    comp = U.T @ c_star
    mask = s >= s[0] / STRONG_BAND_CUTOFF
    frac = float((comp[mask] ** 2).sum() / (c_star @ c_star))
    pre_std = Vt.T[:, mask] @ (comp[mask] / s[mask])
    v_pre = pre_std * xsd  # unstandardize to raw context coords

    rng = np.random.default_rng(SEED)
    v_rand = rng.standard_normal(d)
    v_rand *= np.linalg.norm(v_pre) / np.linalg.norm(v_rand)

    stats = {
        "n": n,
        "d": d,
        "lambda": lam,
        "strong_band_cutoff": f"sigma_max/{STRONG_BAND_CUTOFF:g}",
        "n_band_dirs": int(mask.sum()),
        "reachable_cstar_energy_frac": frac,
        "norm_c_star": float(np.linalg.norm(c_star)),
        "norm_dy": float(np.linalg.norm(dy)),
        "norm_v_pre": float(np.linalg.norm(v_pre)),
        "norm_v_rand": float(np.linalg.norm(v_rand)),
        "predicted_reachable_magnitude": float(np.sqrt(frac) * np.linalg.norm(c_star)),
        "cos_dy_c_star": float(dy @ c_star / (np.linalg.norm(dy) * np.linalg.norm(c_star))),
        "rand_seed": SEED,
    }
    # Parity vs the committed preimage_check.json (same data + recipe => tight).
    for key, got in (
        ("c_star", stats["norm_c_star"]),
        ("dy", stats["norm_dy"]),
        ("v_pre", stats["norm_v_pre"]),
        ("frac", frac),
    ):
        assert abs(got - EXPECTED[key]) < 0.02 * max(1.0, EXPECTED[key]), (
            f"parity vs preimage_check.json failed for {key}: {got:.4f} vs {EXPECTED[key]}"
        )
    vecs = {
        "c_star": c_star.astype(np.float32),
        "dy": dy.astype(np.float32),
        "v_pre": v_pre.astype(np.float32),
        "v_rand": v_rand.astype(np.float32),
    }
    return vecs, stats


def _read_sharded_jsonl(stage_dir: Path, hf_dir: str, stem: str) -> list[dict]:
    """Stage + parse a possibly line-sharded rollout JSONL (upload-policy shards)."""
    from explore_persona_space.orchestrate import hub

    from huggingface_hub.errors import EntryNotFoundError

    single = stage_dir / f"{stem}.jsonl"
    manifest = stage_dir / f"{stem}.manifest.json"
    rows: list[dict] = []
    try:
        hub.stage_hub_file(C.HF_DATA_REPO, f"{hf_dir}/{stem}.manifest.json", manifest)
    except EntryNotFoundError:
        names = []  # unsharded payload: no manifest, single {stem}.jsonl
    else:
        names = [m["name"] for m in json.load(open(manifest))["shards"]]
    files: list[Path]
    if names:
        files = [hub.stage_hub_file(C.HF_DATA_REPO, f"{hf_dir}/{n}", stage_dir / n) for n in names]
    else:
        files = [hub.stage_hub_file(C.HF_DATA_REPO, f"{hf_dir}/{stem}.jsonl", single)]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def build_probe_inputs(out_path: Path) -> dict:
    """256 fixed-seed single-turn intersection rows + BASE answer texts."""
    from explore_persona_space.orchestrate import hub

    ctx = LadderContext(LADDER_OUT_ROOT, ["B"], layer=LAYER_STAR)
    idx = ctx.corpora[CORPUS]
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(idx.n, size=N_PROBE_ROWS, replace=False))
    sel_rows = [idx.rows[int(i)] for i in sel]
    sel_ids = {r["id"] for r in sel_rows}

    stage = STAGE_ROOT / "hf_dl"
    stage.mkdir(parents=True, exist_ok=True)
    corpus_path = hub.stage_hub_file(
        C.HF_DATA_REPO,
        f"{C.CORPUS_HF_PATH}/{C.CORPUS_SINGLE_FILENAME}",
        stage / C.CORPUS_SINGLE_FILENAME,
    )
    query_of: dict[str, dict] = {}
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r["id"] in sel_ids:
                    query_of[r["id"]] = r
    gen_rows = _read_sharded_jsonl(stage, f"{C.HF_PREFIX}/raw_completions/gen/{CORPUS}", "B")
    answer_of = {r["id"]: r for r in gen_rows if r["id"] in sel_ids}

    missing_q = [r["id"] for r in sel_rows if r["id"] not in query_of]
    missing_a = [r["id"] for r in sel_rows if r["id"] not in answer_of]
    assert not missing_q and not missing_a, (
        f"probe rows missing corpus/answer text: q={missing_q[:3]} a={missing_a[:3]}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sel_rows:
            rec = {
                "id": r["id"],
                "query": query_of[r["id"]]["query"],
                "answer_text": answer_of[r["id"]]["text"],
                "group": r.get("group"),
                "cluster": r.get("cluster"),
                "class": r.get("class"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    flagged = sum(
        1
        for r in sel_rows
        if answer_of[r["id"]].get("truncated") or answer_of[r["id"]].get("repetition_flag")
    )
    return {
        "n_rows": len(sel_rows),
        "row_seed": SEED,
        "corpus": CORPUS,
        "answer_source": "B",
        "answer_seed": C.GEN_SEED,
        "n_flagged_answers": flagged,  # 0 by construction (intersection is unflagged)
        "row_index_source": "ladder CorpusIndex store order (intersection manifest)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=STAGE_ROOT / "inputs")
    ap.add_argument("--skip-upload", action="store_true", help="build only (local)")
    args = ap.parse_args()

    from explore_persona_space.orchestrate import hub
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    vecs, stats = fit_and_vectors()
    for name, arr in vecs.items():
        np.save(out / f"{name}.npy", arr)
    print(f"[vectors] saved 4 vectors -> {out} ({stats['norm_v_pre']=:.3f})", flush=True)

    probe_stats = build_probe_inputs(out / "probe_inputs.jsonl")
    print(f"[vectors] probe_inputs.jsonl rows={probe_stats['n_rows']}", flush=True)

    pins = json.load(open(PROJECT_ROOT / "eval_results/issue_1902/revision_pins.json"))
    files = sorted(p.name for p in out.iterdir() if p.name != "meta.json")
    meta = {
        "round": "steer_probe (user-chat inline override, task #1902)",
        "pair": "B->S",
        "layer": LAYER_STAR,
        "corpus": CORPUS,
        "arm": ARM,
        "vector_stats": stats,
        "probe_inputs": probe_stats,
        "revision_pins": {"B": pins["B"]},
        "model_id_B": C.MODEL_IDS["B"],
        "files_sha256": {n: _sha256(out / n) for n in files},
        "metadata": {
            **as_metadata_dict(git_provenance()),
            "script": "scripts/issue1902_steer_vectors.py",
        },
    }
    R._write_json_atomic(out / "meta.json", meta)

    if args.skip_upload:
        print("[vectors] --skip-upload: done (local only)", flush=True)
        sys.exit(0)

    api = R._hf_api()
    hub.assert_hub_dir_filecounts(str(out), INPUTS_HF_DIR)  # deterministic guard, outside retry
    hub.retry_transient(
        lambda: api.upload_folder(
            folder_path=str(out),
            path_in_repo=INPUTS_HF_DIR,
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message="issue1902 steer_probe: P1 vectors + probe inputs",
        ),
        what=f"upload_folder {INPUTS_HF_DIR}",
    )
    expected = [f"{INPUTS_HF_DIR}/{n}" for n in [*files, "meta.json"]]
    missing = hub.verify_repo_paths_uploaded(
        api, C.HF_DATA_REPO, expected, path_in_repo=INPUTS_HF_DIR, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"steer_probe inputs upload verify FAILED, missing: {missing}")
    print(f"[vectors] uploaded + verified {len(expected)} files -> {INPUTS_HF_DIR}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
