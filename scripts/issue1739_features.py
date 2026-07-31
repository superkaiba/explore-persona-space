"""Text-control feature builder for issue #1739 arms 15/16 (round-2 C3).

Builds the injected inputs the fits CLI consumes via ``--text-emb`` /
``--text-features``:

- ``emb``: hashed token-frequency vectors of the CONTEXT text (plan §5
  "TF-IDF or token-frequency" — the hashing trick keeps this dependency-free;
  L2-normalized counts over a fixed hash dimension).
- ``features``: trivial surface statistics of the context + the mean
  rollout-response length per context (plan §5 length/lexical control;
  perplexity is OMITTED — it needs a model forward and is recorded as a
  deviation in the run provenance).

CONTENT HYGIENE (binding): contexts/rollouts come from harmful-content /
real-user corpora. This script reads text fields IN PROCESS only and never
prints/logs text — ids + counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue1739_features.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

HASH_DIM = 256


def _context_text(row: dict) -> str:
    """Context text = prefix + query (the staged-context field names)."""
    parts = [
        str(row.get(k) or "") for k in ("prefix_text", "query", "context_text", "question", "text")
    ]
    return "\n".join(p for p in parts if p)


def hashed_token_freq(text: str, dim: int = HASH_DIM):
    """L2-normalized hashed token-frequency vector (dependency-free TF)."""
    import numpy as np

    v = np.zeros(dim, dtype=np.float64)
    for tok in text.lower().split():
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "little")
        v[h % dim] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def surface_features(text: str, mean_resp_len: float):
    """Surface stats: lengths, lexical fractions, mean rollout response length."""
    import numpy as np

    n_chars = len(text)
    words = text.split()
    n_words = len(words)
    return np.array(
        [
            float(n_chars),
            float(n_words),
            float(np.mean([len(w) for w in words])) if words else 0.0,
            sum(c.isdigit() for c in text) / max(n_chars, 1),
            sum(not c.isalnum() and not c.isspace() for c in text) / max(n_chars, 1),
            text.count("?"),
            float(mean_resp_len),
        ],
        dtype=np.float64,
    )


def build_features(contexts_jsonls: list[Path], rollout_dir: Path | None, out_path: Path) -> dict:
    """Compose the npz ({context_ids, emb, features}) the fits CLI injects."""
    import numpy as np

    resp_len: dict[str, list[int]] = {}
    if rollout_dir is not None and rollout_dir.exists():
        for p in sorted(rollout_dir.rglob("*.json")):
            try:
                payload = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            cid = payload.get("context_id")
            if cid is None:
                continue
            comp = payload.get("completion")
            if isinstance(comp, str):
                resp_len.setdefault(str(cid), []).append(len(comp))
    ids: list[str] = []
    embs, feats = [], []
    seen: set[str] = set()
    for path in contexts_jsonls:
        for line in path.open(encoding="utf-8"):  # text-mode iteration (never splitlines)
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row.get("context_id"))
            if cid in seen:
                continue
            seen.add(cid)
            text = _context_text(row)
            ids.append(cid)
            embs.append(hashed_token_freq(text))
            lens = resp_len.get(cid, [])
            feats.append(surface_features(text, float(np.mean(lens)) if lens else 0.0))
    if not ids:
        raise SystemExit(f"[features] zero contexts found in {len(contexts_jsonls)} jsonl files")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")  # np.savez suffix trap (#1092)
    with tmp.open("wb") as fh:
        np.savez(
            fh,
            context_ids=np.asarray(ids),
            emb=np.stack(embs).astype(np.float32),
            features=np.stack(feats).astype(np.float32),
        )
    tmp.replace(out_path)
    print(
        f"[features] wrote {out_path}: n_contexts={len(ids)} emb_dim={HASH_DIM} "
        f"n_surface={feats[0].shape[0]} n_with_resp_len={sum(1 for i in ids if i in resp_len)}",
        flush=True,
    )
    return {"n_contexts": len(ids), "path": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--contexts-jsonl", type=Path, nargs="+", required=True)
    ap.add_argument("--rollout-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    files: list[Path] = []
    for pattern in args.contexts_jsonl:
        if pattern.exists():
            files.append(pattern)
        else:  # glob form passed quoted
            files.extend(sorted(pattern.parent.glob(pattern.name)))
    if not files:
        raise SystemExit(f"[features] no contexts jsonl matched {args.contexts_jsonl}")
    build_features(files, args.rollout_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
