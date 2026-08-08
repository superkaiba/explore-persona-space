"""#1776 p04_pairs phase: J-pair manifest build + sharded capture + merge.

Committed-by-path port of the three ``uv run python - <<'PY'`` heredocs that
previously lived inline in ``issue1776_dispatch.sh`` (crash-fix round for
att-20260729-060640: inline heredocs are un-lintable, un-smokeable, and their
stderr died with the instance — project rule: committed scripts invoked by
path, never inline interpreter bodies). Logic ported verbatim; only the argv
plumbing changed.

Subcommands:
  build          J-pair manifest (plan §4 P0.4): seeded sample of
                 LMSYS-provenance rows from the n1m TRAIN pool, text joined
                 from the #779 raw_completions chunks at the pin; G-PARITY
                 exclusion list applied at the corpus.
  capture-shard  Sharded teacher-forced capture of the J pairs via the
                 PRODUCER's rig (parity.recompute_row -> issue779_collect
                 capture fns): cx_last@{14,19} + v@19. GPU-bound.
  merge          Merge capture shards back into manifest order ->
                 jpair_capture.pt + v_pool.pt + acts14/acts19.

Content hygiene: row text is never printed; reports carry counts + shas only.
Exit contract: 0 PASS / nonzero fail-loud; explicit sys.exit before
interpreter finalization (torch/transformers atexit race — gotchas.md #1689).
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before torch import: thread caps freeze at torch import

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1776_common as C76  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL written with ensure_ascii json.dumps (text-safe split)."""
    return [json.loads(x) for x in path.read_text().split("\n") if x.strip()]


def cmd_build(args: argparse.Namespace) -> int:
    """Build jpairs.jsonl: seeded chunk-permutation lazy download from the pin."""
    import issue779_ffc_n1m_generate_capture as N1G
    from huggingface_hub import HfApi

    n_pairs, out_dir, manifest_dir = args.n_pairs, args.out_dir, args.manifest_dir
    exclusion_path = args.exclusion
    assert exclusion_path.exists(), (
        f"G-PARITY exclusion list missing (p0_parity ran?): {exclusion_path}"
    )
    excluded_ci = {int(r["ci"]) for r in json.loads(exclusion_path.read_text())["excluded"]}
    out_path = out_dir / "jpairs.jsonl"
    if out_path.exists():
        rows = _read_jsonl(out_path)
        if len(rows) == n_pairs:
            print(f"[jpairs] resume: {out_path} already has {n_pairs} pairs; skip")
            return 0
    pin = C76.resolve_data_repo_pin()
    pool, _meta = N1G.read_manifest_pool(manifest_dir)
    lmsys_ci = {int(r["i"]) for r in pool if r.get("corpus") == "lmsys"}
    assert lmsys_ci, "no lmsys rows in the sampling manifest"

    api = HfApi()
    base = "issue779_monitoring/fitter-fair-comparison-n1m/raw_completions"
    files = sorted(
        f
        for f in hub.list_hf_files_under_path(
            api, C76.HF_DATA_REPO, base, repo_type="dataset", revision=pin
        )
        if f.endswith(".json") and "skipped" not in Path(f).name
    )
    assert files, f"no raw chunks under {base}@{pin}"
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    kept: list[dict] = []
    seen: set[int] = set()
    cache = out_dir / "raw_chunks"
    n_chunks_used = 0
    for oi in order:
        f = files[int(oi)]
        local = hub.stage_hub_file(
            C76.HF_DATA_REPO, f, cache / Path(f).name, repo_type="dataset", revision=pin
        )
        n_chunks_used += 1
        for r in json.loads(Path(local).read_text())["rows"]:
            ci = int(r["ci"])
            if ci in excluded_ci:
                continue  # plan §3: failed-parity rows leave BOTH arms' corpus
            if ci in lmsys_ci and ci not in seen and r.get("response"):
                seen.add(ci)
                kept.append(
                    {
                        "pair_id": f"ci{ci}",
                        "prompt": r["prompt"],
                        "response": r["response"],
                        "ci": ci,
                        "chunk": Path(f).name,
                    }
                )
        if len(kept) >= n_pairs:
            break
    assert len(kept) >= n_pairs, f"only {len(kept)} lmsys pairs across {n_chunks_used} chunks"
    idx = np.sort(rng.choice(len(kept), size=n_pairs, replace=False))
    sel = [kept[int(i)] for i in idx]
    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in sel))
    tmp.replace(out_path)
    C76.atomic_write_json(
        out_dir / "jpairs_build_report.json",
        {
            "n_pairs": n_pairs,
            "n_chunks_visited": n_chunks_used,
            "n_lmsys_pool": len(lmsys_ci),
            "n_parity_excluded_ci": len(excluded_ci),
            "pin": pin,
            "seed": 0,
            "note": "seeded chunk-permutation lazy download; sample restricted to visited chunks",
            "jpairs_sha256": sha256(out_path.read_bytes()).hexdigest(),
            "repro": C76.repro_meta(),
        },
    )
    print(f"[jpairs] wrote {n_pairs} pairs from {n_chunks_used} chunks -> {out_path}")
    return 0


def cmd_capture_shard(args: argparse.Namespace) -> int:
    """Teacher-forced capture of one J-pair shard via parity.recompute_row (GPU)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import issue779_common as C
    import issue1776_parity as PAR

    pairs_path, out_path = args.pairs, args.out
    shard, n_shards = args.shard_index, args.num_shards
    rows = _read_jsonl(pairs_path)
    rows = rows[shard::n_shards]
    if out_path.exists():
        have = torch.load(out_path, map_location="cpu", weights_only=True)
        if list(have["pair_id"]) == [r["pair_id"] for r in rows]:
            print(f"[p04-capture] shard {shard}: {out_path} complete; skip")
            return 0
    fields = PAR.consumed_fields(C76.SOURCE_LAYER, C76.READOUT_LAYER)
    tok = AutoTokenizer.from_pretrained(C.DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        C.DEFAULT_MODEL, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    ids, v19, c14, c19 = [], [], [], []
    for j, r in enumerate(rows):
        vec = PAR.recompute_row(model, tok, r["prompt"], r["response"], fields)
        ids.append(r["pair_id"])
        c14.append(vec[("cx_last", C76.SOURCE_LAYER)].to(torch.float32).cpu())
        c19.append(vec[("cx_last", C76.READOUT_LAYER)].to(torch.float32).cpu())
        v19.append(vec[("v_x", C76.READOUT_LAYER)].to(torch.float32).cpu())
        if (j + 1) % 32 == 0:
            print(f"[p04-capture] shard {shard}: {j + 1}/{len(rows)}", flush=True)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "pair_id": ids,
            "v19": torch.stack(v19),
            "c14": torch.stack(c14),
            "c19": torch.stack(c19),
            "layers": [C76.SOURCE_LAYER, C76.READOUT_LAYER],
        },
        tmp,
    )
    tmp.replace(out_path)
    print(f"[p04-capture] shard {shard}: {len(ids)} pairs -> {out_path}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge capture shards into manifest order -> jpair_capture/v_pool/acts."""
    jdir, n_shards = args.jpairs_dir, args.num_shards
    rows = _read_jsonl(jdir / "jpairs.jsonl")
    by_id: dict[str, tuple] = {}
    for g in range(n_shards):
        sh = torch.load(jdir / f"cap_shard{g}.pt", map_location="cpu", weights_only=True)
        for i, pid in enumerate(sh["pair_id"]):
            by_id[pid] = (sh["v19"][i], sh["c14"][i], sh["c19"][i])
    missing = [r["pair_id"] for r in rows if r["pair_id"] not in by_id]
    assert not missing, f"capture shards missing {len(missing)} pairs (e.g. {missing[:3]})"
    order = [r["pair_id"] for r in rows]
    v = torch.stack([by_id[p][0] for p in order])
    c14 = torch.stack([by_id[p][1] for p in order])
    c19 = torch.stack([by_id[p][2] for p in order])
    torch.save(
        {"pair_id": order, "v19": v, "c14": c14, "c19": c19, "layers": [14, 19]},
        jdir / "jpair_capture.pt",
    )
    torch.save({"v": v}, jdir / "v_pool.pt")
    torch.save(c14, jdir / "acts14.pt")
    torch.save(c19, jdir / "acts19.pt")
    print(f"[p04-merge] {v.shape[0]} pairs merged (H={v.shape[1]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the J-pair manifest (jpairs.jsonl)")
    b.add_argument("--n-pairs", type=int, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--manifest-dir", type=Path, required=True)
    b.add_argument("--exclusion", type=Path, required=True, help="G-PARITY exclusion_list.json")
    b.set_defaults(fn=cmd_build)

    c = sub.add_parser("capture-shard", help="teacher-forced capture of one shard (GPU)")
    c.add_argument("--pairs", type=Path, required=True, help="jpairs.jsonl")
    c.add_argument("--out", type=Path, required=True, help="cap_shard<g>.pt target")
    c.add_argument("--shard-index", type=int, required=True)
    c.add_argument("--num-shards", type=int, required=True)
    c.set_defaults(fn=cmd_capture_shard)

    m = sub.add_parser("merge", help="merge capture shards into manifest order")
    m.add_argument("--jpairs-dir", type=Path, required=True)
    m.add_argument("--num-shards", type=int, required=True)
    m.set_defaults(fn=cmd_merge)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
