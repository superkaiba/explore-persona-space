"""Select an outcome-blind, source-stratified endpoint panel for the K5 pilot."""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from issue1092_gpu_phase import (  # noqa: E402
    INSTRUCT_MODEL,
    INSTRUCT_REVISION,
    PRETRAINED_MODEL,
    PRETRAINED_REVISION,
    _render_full_conversation,
)
from transformers import AutoTokenizer  # noqa: E402


def select_stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Sample fixed proportional source counts without replacement."""
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} eligible conversations for requested {n}")
    sources = sorted({r["source"] for r in rows})
    counts = np.array([sum(r["source"] == source for r in rows) for source in sources])
    quotas = n * counts / counts.sum()
    sizes = np.floor(quotas).astype(int)
    for i in np.argsort(-(quotas - sizes), kind="stable")[: n - sizes.sum()]:
        sizes[i] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for source, size in zip(sources, sizes, strict=True):
        group = sorted((r for r in rows if r["source"] == source), key=lambda r: r["conv_id"])
        chosen.extend(group[i] for i in rng.choice(len(group), int(size), replace=False))
    return sorted(chosen, key=lambda r: r["conv_id"])


def main() -> None:
    """Tokenize only logged prefixes; select and fingerprint the common panel."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-panel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=1000)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite selection: {args.out}")
    specs = {
        "instruct": (INSTRUCT_MODEL, INSTRUCT_REVISION),
        "pretrained": (PRETRAINED_MODEL, PRETRAINED_REVISION),
    }
    toks = {
        key: AutoTokenizer.from_pretrained(model, revision=rev)
        for key, (model, rev) in specs.items()
    }
    inputs = sorted(args.parent_panel.glob("panel_armR_shard*.jsonl"))
    if len(inputs) != 10:
        raise ValueError("expected all ten original panel shards")
    rows = []
    for path in inputs:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if len(rows) != 5000 or len({r["id"] for r in rows}) != 5000:
        raise ValueError("original panel must contain 5,000 unique conversations")
    eligible, excluded = [], []
    # 6080 + doubled answer cap 2048 + explicit 64-token rendering allowance = 8192.
    prompt_bound = 6080
    for row in rows:
        turns = row["turns"]
        positions = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
        if len(positions) < 12:
            raise ValueError("original panel contains a shallow conversation")
        lengths = {}
        for turn in (1, 12):
            history = turns[: positions[turn - 1]]
            for model, tok in toks.items():
                if model == "instruct":
                    prompt = tok.apply_chat_template(
                        history, tokenize=False, add_generation_prompt=True
                    )
                else:
                    prompt = _render_full_conversation(history, "pretrained") + "\n\nAssistant:"
                lengths[f"{model}_turn{turn}"] = len(tok.encode(prompt, add_special_tokens=False))
        item = {
            "conv_id": str(row["id"]),
            "source": row["source"],
            "turns": turns[: positions[11] + 1],
            "prompt_lengths": lengths,
        }
        if max(lengths.values()) <= prompt_bound:
            eligible.append(item)
        else:
            excluded.append(
                {"conv_id": item["conv_id"], "source": item["source"], "prompt_lengths": lengths}
            )
    chosen = select_stratified(eligible, args.n, 0)
    args.out.mkdir(parents=True)
    panel = args.out / "panel.jsonl"
    panel.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in chosen))
    manifest = {
        "status": "complete",
        "parent_repo": "superkaiba1/explore-persona-space-data",
        "parent_revision": "f70b746317b59363e70a4b270316703b8360d6f4",
        "parent_prefix": "issue825_userbase_map/analysis_tensors/turn_dynamics/panel",
        "parent_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        },
        "parent_n": len(rows),
        "eligible_n": len(eligible),
        "selected_n": len(chosen),
        "source_counts_parent": dict(Counter(r["source"] for r in rows)),
        "source_counts_eligible": dict(Counter(r["source"] for r in eligible)),
        "source_counts_selected": dict(Counter(r["source"] for r in chosen)),
        "selection_seed": 0,
        "selection": "proportional source quotas; uniform without replacement within source",
        "prompt_token_bound": prompt_bound,
        "capture_window": 8192,
        "reserved_answer_cap": 2048,
        "reserved_rendering_allowance": 64,
        "panel_sha256": hashlib.sha256(panel.read_bytes()).hexdigest(),
        "selected_ids": [r["conv_id"] for r in chosen],
        "excluded_before_generation": excluded,
        "model_specs": specs,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("eligible_n", "selected_n", "source_counts_selected", "panel_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
