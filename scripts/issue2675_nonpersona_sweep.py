"""Issue #2675 follow-up: strength x rollout sweep of library non-persona neurons.

For every neuron-direction in ``neurons.json`` (from
``issue2675_nonpersona_select.py``) and every strength alpha in ALPHAS, ask
Qwen3-8B "Who are you?" once greedily and N_SAMPLED times with sampling.

Strength uses the Neuron Persona Library scale: the push adds
``c * W_down[:, n]`` to the residual after layer L with
``c = sign * alpha * r_L / ||W_down[:, n]||``, where r_L is the median residual
norm after layer L over the "Who are you?" prompt tokens (chat template,
thinking off, first token excluded). The hook from issue2675_neuron_steer adds
c at ``down_proj`` input index n, which is the same intervention.

Phases (checkpointed under ``--out-root``, resumable per batch):
  rnorm   r_L per layer on the prompt + W_down column norms for picked neurons
  sweep   greedy pass, then sampled pass (unsteered baseline included)

Rows land in ``<out_root>/raw_completions/nonpersona/batch_NNNNN.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2675_neuron_steer as base  # noqa: E402
import torch  # noqa: E402

PROMPT = "Who are you?"
ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
N_SAMPLED = 20
PHASE = "nonpersona"
QUESTION = base.Question("self_heldout_00", PROMPT, "self_heldout", "self")
assert base.SELF_HELDOUT[0] == PROMPT


def picked_key(e: dict) -> str:
    """Stable name for one neuron-direction."""
    return f"L{e['layer']}.N{e['neuron']}.{e['direction']}"


@torch.no_grad()
def phase_rnorm(model, tok, picks: list[dict], out_root: Path) -> dict:
    """Library r_L (median residual norm after layer L, prompt tokens, token 0 excluded)."""
    enc = tok([base.chat_text(tok, PROMPT)], return_tensors="pt").to(model.device)
    hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
    n_layers = model.config.num_hidden_layers
    assert len(hs) == n_layers + 1, len(hs)
    r = {}
    for L in range(n_layers):
        nrm = hs[L + 1][0].float().norm(dim=-1)
        assert nrm.shape == (enc["input_ids"].shape[1],), nrm.shape
        r[L] = float(nrm[1:].median())
    col = {}
    for e in picks:
        W = model.model.layers[e["layer"]].mlp.down_proj.weight
        col[picked_key(e)] = float(W[:, e["neuron"]].float().norm())
    res = {
        "prompt_tokens": int(enc["input_ids"].shape[1]),
        "r_L": {str(L): v for L, v in r.items()},
        "w_down_col_norm": col,
    }
    base.write_json(out_root / "rnorm.json", res)
    base.log(f"[rnorm] prompt_tokens={res['prompt_tokens']} r_8={r.get(8, float('nan')):.2f}")
    return res


def sweep_jobs(picks: list[dict], rn: dict, n_sampled: int, alphas) -> tuple[list, list]:
    """Greedy and sampled job lists; each job's cond carries the hook alpha c."""
    conds = [base.Condition("baseline", "baseline", None, None, None, 0.0, 0.0)]
    meta = {"baseline": {"set": "baseline"}}
    for e in picks:
        k = picked_key(e)
        sign = 1.0 if e["direction"] == "up" else -1.0
        cn = rn["w_down_col_norm"][k]
        rL = rn["r_L"][str(e["layer"])]
        for a in alphas:
            c = sign * a * rL / cn
            cid = f"{k}@{a:g}"
            conds.append(base.Condition(cid, e["set"], k, e["layer"], e["neuron"], c, sign * a))
            meta[cid] = {k2: e[k2] for k2 in ("set", "lib_group", "lib_name", "lib_strength")}
    greedy = [base.Job(f"{c.cond_id}|greedy|0", c, QUESTION, 0) for c in conds]
    sampled = [
        base.Job(f"{c.cond_id}|sample|{s}", c, QUESTION, s) for s in range(n_sampled) for c in conds
    ]
    return greedy + sampled, [meta]


@torch.no_grad()
def generate(model, tok, steerer, jobs, greedy: bool, max_new_tokens: int, seed: int):
    """One completion per job with per-row steering; greedy or Qwen-recommended sampling."""
    enc = tok([base.chat_text(tok, j.q.text) for j in jobs], return_tensors="pt", padding=True)
    enc = enc.to(model.device)
    steerer.set_rows([j.cond for j in jobs])
    torch.manual_seed(seed)
    kw = (
        {"do_sample": False}
        if greedy
        else {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    )
    try:
        out = model.generate(
            **enc, **kw, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id
        )
    finally:
        steerer.clear()
    new = out[:, enc["input_ids"].shape[1] :].tolist()
    stop = base.eos_ids(model, tok)
    rows = []
    for j, ids in zip(jobs, new, strict=True):
        cut = next((i for i, t in enumerate(ids) if t in stop), None)
        kept = ids if cut is None else ids[:cut]
        rows.append(
            {
                "job_id": j.job_id,
                **asdict(j.cond),
                "decode": "greedy" if greedy else "sample",
                "qid": j.q.qid,
                "question": j.q.text,
                "group": j.q.group,
                "kind": j.q.kind,
                "sample": j.sample,
                "text": tok.decode(kept, skip_special_tokens=True),
                "n_new_tokens": len(kept),
                "finish_reason": "stop" if cut is not None else "length",
                "seed": seed,
            }
        )
    return rows


def run_sweep(jobs, meta: dict, model, tok, args, out_root: Path) -> None:
    """Generate all jobs in fixed batches (greedy batches first); skip batches on disk."""
    pdir = out_root / "raw_completions" / PHASE
    pdir.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha256(
        "\n".join(f"{j.job_id}:{j.cond.alpha:.6f}" for j in jobs).encode()
        + f"|{args.batch_size}|{args.max_new_tokens}".encode()
    ).hexdigest()
    man_path = pdir / "manifest.json"
    if man_path.exists():
        assert json.loads(man_path.read_text())["fingerprint"] == fp, "job list changed"
    else:
        base.write_json(man_path, {"fingerprint": fp, "n_jobs": len(jobs)})
    greedy = [j for j in jobs if "|greedy|" in j.job_id]
    sampled = [j for j in jobs if "|sample|" in j.job_id]
    assert len(greedy) + len(sampled) == len(jobs)
    bs = args.batch_size
    batches = [(True, greedy[i : i + bs]) for i in range(0, len(greedy), bs)]
    batches += [(False, sampled[i : i + bs]) for i in range(0, len(sampled), bs)]
    layers = {j.cond.layer for j in jobs if j.cond.layer is not None}
    steerer = base.NeuronSteerer(model, sorted(layers))
    t0 = time.time()
    n_new = 0
    for b, (is_greedy, bj) in enumerate(batches):
        path = pdir / f"batch_{b:05d}.jsonl"
        if path.exists():
            continue
        rows = generate(
            model, tok, steerer, bj, is_greedy, args.max_new_tokens, base.SEED * 1000 + b
        )
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for r in rows:
                r["batch"] = b
                r.update(meta[r["cond_id"]])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        n_new += 1
        cap = sum(r["finish_reason"] == "length" for r in rows) / len(rows)
        base.log(
            f"[{PHASE}] unit {b + 1}/{len(batches)} batch_{b:05d} "
            f"{'greedy' if is_greedy else 'sample'} elapsed={time.time() - t0:.0f}s cap_hit={cap:.2f}"
        )
        if n_new % args.upload_every == 0:
            base.upload(out_root, args)
    for h in steerer.handles:
        h.remove()
    base.write_json(pdir / "done.json", {"fingerprint": fp, "n_batches": len(batches)})


def _import_check() -> None:
    """Static argparse-attribute + helper-call-shape check, then exit."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    raise SystemExit(0)


def main() -> None:
    """Parse args and run the requested phases."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--neurons", required=True, help="neurons.json from the select script")
    ap.add_argument("--phases", default="rnorm,sweep")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--run-name", default="nonpersona_sweep")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--upload-every", type=int, default=10)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        _import_check()

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    sel = json.loads(Path(args.neurons).read_text())
    picks = sel["neurons"]
    model_id = base.SMOKE_MODEL_ID if args.smoke else base.MODEL_ID
    tok, model = base.load_model(model_id, args.device)
    n_int, n_layers = model.config.intermediate_size, model.config.num_hidden_layers
    n_sampled, alphas = N_SAMPLED, ALPHAS
    if args.smoke:
        picks = [
            {**e, "layer": e["layer"] % n_layers, "neuron": e["neuron"] % n_int}
            for e in (picks[:2] + picks[-2:])
        ]
        n_sampled, alphas = 2, ALPHAS[2:4]
        args.max_new_tokens = min(args.max_new_tokens, 12)
    else:
        assert (n_int, n_layers) == (12288, 36), (n_int, n_layers)
    base.write_json(
        out_root / "run_config.json",
        {
            "model_id": model_id,
            "args": vars(args),
            "prompt": PROMPT,
            "alphas": list(alphas),
            "n_sampled": n_sampled,
            "library_revision": sel["library_revision"],
            "neurons_csv_sha256": sel["csv_sha256"],
            "counts": sel["counts"],
            "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
            "torch": torch.__version__,
            **as_metadata_dict(git_provenance(), phase="nonpersona-generate"),
        },
    )
    base.write_json(out_root / "neurons.json", {**sel, "neurons": picks})
    phases = args.phases.split(",")
    base.log(f"[main] model={model_id} phases={phases} picks={len(picks)} out_root={out_root}")
    if "rnorm" in phases:
        phase_rnorm(model, tok, picks, out_root)
        base.upload(out_root, args)
    if "sweep" in phases:
        rn = json.loads((out_root / "rnorm.json").read_text())
        jobs, (meta,) = sweep_jobs(picks, rn, n_sampled, alphas)
        base.log(f"[{PHASE}] {len(jobs)} jobs")
        run_sweep(jobs, meta, model, tok, args, out_root)
        base.upload(out_root, args)
    base.log("[main] done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
