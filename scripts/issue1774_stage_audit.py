"""#1774 P0 — staging + reuse fitness audit (VM, 0 GPU).

1. Verify the staged #1092 store matches the pinned HF rev (size + sha256
   spot-check at the CONSUMED L14/18/19 files only — never a random dir member;
   plan §4 fact-check); re-stage via ``hub.stage_hub_prefix`` on mismatch.
2. Rebuild parent-parity folds (seed 0, conv_id→prefix_id fallback) + write the
   fold/arm registries with the n/d estimator-validity audit (17,308 ±0 assert).
3. Parent-lineage duty (artifact-reuse check (k)): record the
   origin/main..origin/issue-1092 diff for every imported module.
4. Realized-keys duty (check (c)): shape asserts on consumed store files + r_B.
5. Freeze the P1 draw-context manifest (99 dense-core prefixes × 20 queries +
   trait-stratum rows → 2,216 contexts, seed 1774).

Usage: uv run python scripts/issue1774_stage_audit.py [--smoke] [--out-root D]
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps + .env bind BEFORE the heavy imports below (BLAS/torch
# pools freeze at import time; tests/test_shared_vm_thread_caps.py).
load_dotenv()

import numpy as np  # noqa: E402

import issue1774_common as c  # noqa: E402

LINEAGE_MODULES = [
    "scripts/issue1092_fit_grid.py",
    "scripts/issue1092_partb_operator.py",
    "scripts/issue1092_gpu_phase.py",
    "scripts/issue1092_inline_compose_chain.py",
    "scripts/issue923_fit_decomposition.py",
]

# Duty-(k) resolution recorded at implementation time (round A), verified by the
# live git diff below — the plan asm-9 commits (364aecdb46, 6aa4dddcf3,
# e4802bda51, ae6db521a8) were inspected: the round-8.8/8.9 capture fixes ARE
# on main; the residual branch delta on gpu_phase is comment-only.
DUTY_K_RECORD = (
    "issue1092_gpu_phase.py branch delta = comment-only lint waivers; "
    "fit_grid main AHEAD of branch → not-needed (content-equivalent)"
)

# Consumed store files per plan §4 (spot-check ONLY these — the pinned HF tree
# carries all 28 layers per kind; a random member would trigger spurious re-stage).
CONSUMED_KINDS = ("context_end", "prefix_end", "t1", "t2", "t3")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _consumed_files() -> list[tuple[str, Path]]:
    """(hub-relative path, local path) for the consumed L14/18/19 files."""
    out: list[tuple[str, Path]] = []
    summ = c.summaries_dir()
    for layer in c.LAYERS:
        for kind in CONSUMED_KINDS:
            from issue1092_fit_grid import _summary_shard_paths

            for p in _summary_shard_paths(summ, c.CELL, kind, layer):
                rel = p.relative_to(c.stage_dir())
                out.append((f"{c.STORE_PREFIX}/{rel}", p))
        bare_root = summ / "bare_instruct"
        for p in sorted(bare_root.glob(f"c_q_bare_L{layer:02d}*.npy")):
            rel = p.relative_to(c.stage_dir())
            out.append((f"{c.STORE_PREFIX}/{rel}", p))
    out.append((f"{c.STORE_PREFIX}/corpus/manifest.jsonl", c.manifest_path()))
    # P1/P3 prompt rendering (_render_rows -> issue1092_gpu_phase.load_store) consumes
    # the prefix/query stores too; a partially-staged tree (manifest present, stores
    # absent) must FAIL the audit here, not crash mid-phase (found by the P3 smoke).
    for store in ("prefix_store.jsonl", "query_store.jsonl"):
        out.append((f"{c.STORE_PREFIX}/corpus/{store}", c.stage_dir() / "corpus" / store))
    return out


def _mirror_root() -> Path:
    """The dest root for ``stage_hub_prefix``'s VERBATIM PREFIX MIRROR (#1402).

    ``stage_hub_prefix`` lands every file at ``dest / <repo-relative path>`` —
    passing the FINAL consumed path as dest nests the whole hub prefix under it
    (``corpus/issue1092_realistic_crossing/corpus/manifest.jsonl``), which is the
    att-20260729-033609 GCE P0 crash (artifact-reuse check (h)(iv); the restage
    path never ran on the VM because the store pre-existed there). The mirror
    root must satisfy ``root / STORE_PREFIX == stage_dir()``.
    """
    sd = c.stage_dir()
    assert sd.name == Path(c.STORE_PREFIX).name, (
        f"stage dir {sd} must end with the store-prefix leaf {c.STORE_PREFIX!r} "
        "for the verbatim prefix mirror to land at the consumed layout"
    )
    return sd.parent


def verify_or_stage_store(spot_n: int, apply_restage: bool) -> dict:
    """Per-file size + sha256 spot-check vs get_paths_info at the pinned rev."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = _consumed_files()
    missing = [(rel, p) for rel, p in files if not p.exists()]
    if missing and apply_restage:
        print(f"[stage] {len(missing)} consumed files missing; staging from HF @ {c.STORE_REV}")
        root = _mirror_root()  # dest/<repo-relative path> == stage_dir()/<local rel>
        for prefix in (
            f"{c.STORE_PREFIX}/analysis_tensors/summaries/{c.CELL}",
            f"{c.STORE_PREFIX}/analysis_tensors/summaries/bare_instruct",
            f"{c.STORE_PREFIX}/corpus",
        ):
            hub.stage_hub_prefix(
                c.DATA_REPO, prefix, root, repo_type="dataset", revision=c.STORE_REV
            )
        files = _consumed_files()
        missing = [(rel, p) for rel, p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} consumed store files missing: {missing[:3]}")
    # fix-engaged signal (att-20260729-033609): only reachable once the restage
    # lands the files at the CONSUMED layout (pre-fix, the raise above fired here).
    print(f"[stage] consumed-file set verified: {len(files)} files under {c.stage_dir()}")

    # sha spot-check: deterministic sample of spot_n consumed files
    rng = np.random.default_rng(0)
    idx = rng.choice(len(files), size=min(spot_n, len(files)), replace=False)
    api = HfApi()
    checked, mismatched = [], []
    rels = [files[int(i)][0] for i in idx]
    infos = {
        i.path: i
        for i in hub.retry_transient(
            lambda: api.get_paths_info(
                c.DATA_REPO, rels, repo_type="dataset", revision=c.STORE_REV
            ),
            what="p0 get_paths_info spot-check",
        )
    }
    for i in idx:
        rel, p = files[int(i)]
        info = infos.get(rel)
        if info is None:
            mismatched.append({"path": rel, "reason": "absent-at-pinned-rev"})
            continue
        size_local = p.stat().st_size
        size_hub = getattr(info, "size", None)
        row = {"path": rel, "size_local": size_local, "size_hub": size_hub}
        lfs = getattr(info, "lfs", None)
        if lfs is not None and getattr(lfs, "sha256", None):
            row["sha256_local"] = _sha256(p)
            row["sha256_hub"] = lfs.sha256
            if row["sha256_local"] != lfs.sha256:
                mismatched.append(row)
        elif size_hub is not None and size_local != size_hub:
            mismatched.append(row)
        checked.append(row)
    if mismatched:
        raise RuntimeError(
            f"store spot-check mismatch vs pinned rev {c.STORE_REV}: {mismatched} "
            "(kill criterion: re-stage once via --apply-restage; fail if it persists)"
        )
    return {"n_files": len(files), "n_spot_checked": len(checked), "spot": checked}


def lineage_duty_k() -> dict:
    """Check (k) leg A: origin/main..origin/issue-1092 diff per imported module."""
    subprocess.run(
        ["git", "fetch", "origin", "issue-1092", "--quiet"],
        cwd=c.PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    out: dict[str, dict] = {}
    for mod in LINEAGE_MODULES:
        r = subprocess.run(
            ["git", "log", "--oneline", "origin/main..origin/issue-1092", "--", mod],
            cwd=c.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            out[mod] = {
                "status": "branch-unreadable — leg A unverifiable",
                "stderr": r.stderr[:200],
            }
            continue
        commits = [line for line in r.stdout.strip().split("\n") if line]
        if not commits:
            out[mod] = {"status": "empty-diff (main current)", "commits": []}
            continue
        # inspect content equivalence: diff the module between branch tip and main
        d = subprocess.run(
            ["git", "diff", "--stat", "origin/main...origin/issue-1092", "--", mod],
            cwd=c.PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        body = subprocess.run(
            ["git", "diff", "origin/main...origin/issue-1092", "--", mod],
            cwd=c.PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).stdout
        substantive = [
            line
            for line in body.split("\n")
            if (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            and line[1:].strip()
            and not line[1:].strip().startswith("#")
        ]
        out[mod] = {
            "status": (
                "not-needed (comment-only branch delta)"
                if not substantive
                else "SUBSTANTIVE-DIFF — inspect before reuse"
            ),
            "commits": commits,
            "diffstat": d.stdout.strip(),
            "n_substantive_lines": len(substantive),
        }
    out["_plan_time_record"] = DUTY_K_RECORD
    return out


def realized_keys_duty(smoke: bool) -> dict:
    """Check (c): shape asserts on consumed arrays + r_B keys + jensen npz members."""
    out: dict = {}
    for layer in [c.HEADLINE_LAYER] if smoke else c.LAYERS:
        ctx = c.load_summary_rows(c.CELL, "context_end", layer)
        pfx = c.load_summary_rows(c.CELL, "prefix_end", layer)
        t1 = c.load_summary_rows(c.CELL, "t1", layer)
        assert ctx.shape == (c.EXPECTED_MANIFEST_ROWS, c.HIDDEN_DIM), (layer, ctx.shape)
        assert pfx.shape == ctx.shape and t1.shape == ctx.shape, (pfx.shape, t1.shape)
        bare, q2i = c.load_bare(layer)
        assert bare.shape[0] == c.EXPECTED_BARE_ROWS, (layer, bare.shape)
        assert bare.shape[1] == c.HIDDEN_DIM
        out[f"L{layer}"] = {
            "context_end": list(ctx.shape),
            "prefix_end": list(pfx.shape),
            "t1": list(t1.shape),
            "bare": list(bare.shape),
            "n_bare_query_ids": len(q2i),
        }
    c.stage_rb_bank()
    rb = c.load_rb_bank(c.HEADLINE_LAYER)
    out["rb"] = {t: list(v.shape) for t, v in rb.items()}
    # jensen scalar npz (decides Q1c reuse-vs-refit: refit IS the mainline — plan §4)
    npz_path = (
        c.PROJECT_ROOT
        / "eval_results/issue_1092/inline_mlp_jensen_natural"
        / ("per_prefix_jensen_cell_inst_own.npz")
    )
    if npz_path.exists():
        with np.load(npz_path) as z:
            out["jensen_npz_members"] = sorted(z.files)
            out["jensen_has_gap_vectors"] = "gap_vectors" in z.files
    else:
        out["jensen_npz_members"] = None
        out["jensen_has_gap_vectors"] = False
    return out


def build_registries(smoke: bool) -> tuple[dict, dict, dict]:
    rows = c.load_manifest()
    fit_idx = c.fit_indices(rows)
    if not smoke:
        assert len(fit_idx) == c.EXPECTED_FIT_ROWS, (
            f"battery-excluded fit-arm-A rows {len(fit_idx)} != {c.EXPECTED_FIT_ROWS} ±0"
        )
    else:
        print(f"[smoke] fit rows = {len(fit_idx)} (17,308 assert demoted to log under smoke)")
    banked_idx = c.banked_convention_indices(rows)
    fit_rows = [rows[i] for i in fit_idx]
    folds = c.grouped_folds(fit_rows, len(fit_rows))
    prefix_ids = [str(r.get("prefix_id", "")) for r in fit_rows]
    query_ids = [str(r.get("query_id", "")) for r in fit_rows]
    n_distinct_prefix = len(set(prefix_ids))
    n_distinct_query = len(set(query_ids))
    _bare, q2i = c.load_bare(c.HEADLINE_LAYER)
    missing_q = sorted({q for q in query_ids if q not in q2i})
    assert not missing_q, (
        f"{len(missing_q)} fit-row query_ids missing from bare join: {missing_q[:5]}"
    )

    fold_registry = {
        "meta": c.repro_meta({"script": "scripts/issue1774_stage_audit.py"}),
        "fold_seed": 0,
        "group_key": "conv_id (absent from manifest rows -> prefix_id fallback; "
        "issue1092_fit_grid.py:178 exact reproduction)",
        "n_folds": c.N_FOLDS,
        "fit_arm": c.FIT_ARM,
        "n_fit_rows": len(fit_idx),
        "n_banked_convention_rows": len(banked_idx),
        "fit_manifest_indices": fit_idx,
        "folds": [f.tolist() for f in folds],
    }
    arm_registry = {
        "meta": c.repro_meta(),
        "n_over_d": {
            "arm_context": {
                "n_distinct": len(fit_idx),
                "d": c.HIDDEN_DIM,
                "n_gg_d": len(fit_idx) > 2 * c.HIDDEN_DIM,
            },
            "arm_prefix_end": {
                "n_distinct": n_distinct_prefix,
                "d": c.HIDDEN_DIM,
                "n_gg_d": n_distinct_prefix > 2 * c.HIDDEN_DIM,
            },
            "arm_bare_query": {
                "n_distinct": n_distinct_query,
                "d": c.HIDDEN_DIM,
                "n_gg_d": n_distinct_query > 2 * c.HIDDEN_DIM,
            },
            "arm_query_avg": {
                "n_distinct": n_distinct_prefix,
                "d": c.HIDDEN_DIM,
                "n_gg_d": n_distinct_prefix > 2 * c.HIDDEN_DIM,
            },
        },
        "kernel_claims_arm": "arm_context",  # the only n>>d arm (plan Divergence 2)
        "row_joins": {
            "prefix_ids": prefix_ids,
            "query_ids": query_ids,
        },
    }
    # P1 draw-context manifest: dense-core prefixes × 20 seeded queries + trait stratum
    rng = np.random.default_rng(c.SEED_DRAWS)
    dense = {}
    for i, r in zip(fit_idx, fit_rows, strict=True):
        dense.setdefault(str(r.get("prefix_id")), []).append(i)
    # >= 20 (round 2 Minor): 20 queries are PICKED per core prefix below — a
    # 15-19-row prefix under-fills its pick and the n_contexts == 2216 assert fires.
    core = sorted([p for p, ix in dense.items() if len(ix) >= 20])
    rng.shuffle(core)
    core = core[: (2 if smoke else 99)]
    picked: list[int] = []
    for p in core:
        ix = sorted(dense[p])
        take = rng.choice(len(ix), size=min(2 if smoke else 20, len(ix)), replace=False)
        picked.extend(int(ix[j]) for j in take)
    trait_rows = [i for i, r in enumerate(rows) if r.get("stratum") == "trait_stratum"]
    rng.shuffle(trait_rows)
    n_trait = 1 if smoke else 236
    picked.extend(int(i) for i in trait_rows[:n_trait])
    draws_manifest = {
        "meta": c.repro_meta(),
        "seed": c.SEED_DRAWS,
        "k_draws": 2 if smoke else c.K_DRAWS,
        "n_contexts": len(picked),
        "manifest_indices": sorted(set(picked)),
        "n_core_prefixes": len(core),
        "n_trait_stratum": n_trait,
    }
    if not smoke:
        assert draws_manifest["n_contexts"] == 2216, draws_manifest["n_contexts"]
    return fold_registry, arm_registry, draws_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--spot-n", type=int, default=9)
    ap.add_argument("--apply-restage", action="store_true", help="stage missing files from HF")
    ap.add_argument("--stage-rb", action="store_true", help="stage the pinned r_B bank only")
    args = ap.parse_args(argv)

    if args.stage_rb:
        c.stage_rb_bank()
        print("[p0] r_B bank staged")
        return 0

    out = c.eval_out(args.out_root) / "registry"
    print(f"[phase=p0_stage_audit] stage_dir={c.stage_dir()}")
    store = verify_or_stage_store(args.spot_n, args.apply_restage)
    lineage = lineage_duty_k()
    keys = realized_keys_duty(args.smoke)
    fold_reg, arm_reg, draws_manifest = build_registries(args.smoke)

    c.write_json_atomic(out / "folds.json", fold_reg)
    c.write_json_atomic(out / "arm_registry.json", arm_reg)
    c.write_json_atomic(out / "draws_manifest.json", draws_manifest)
    c.write_json_atomic(
        out / "p0_audit.json",
        {
            "meta": c.repro_meta({"script": "scripts/issue1774_stage_audit.py"}),
            "store_spot_check": store,
            "lineage_duty_k": lineage,
            "realized_keys_duty_c": keys,
            "smoke": bool(args.smoke),
        },
    )
    substantive = {
        m: v
        for m, v in lineage.items()
        if isinstance(v, dict) and "SUBSTANTIVE" in v.get("status", "")
    }
    if substantive:
        print(
            f"[p0] WARNING duty-(k) substantive branch deltas need porting review: "
            f"{sorted(substantive)}"
        )
    print(
        f"[p0] done: fit_rows={fold_reg['n_fit_rows']} folds={len(fold_reg['folds'])} "
        f"draw_contexts={draws_manifest['n_contexts']} spot_checked={store['n_spot_checked']}"
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
