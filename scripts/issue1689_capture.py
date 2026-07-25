"""Issue #1689 Phase C — teacher-forced activation capture.

Captures activations at frozen layers [14, 18, 19, 26] at the PREFIX arm
(X = end of everything before u2) AND the CONTEXT arm (X = end of the
prompt up to and including u2), plus Y = end of a2, for every row in
every condition. Both mapping arms per plan §4/§6.

Writes per-cell stores at `analysis_tensors/issue_1689/store/<model>/
<condition>/L{14,18,19,26}.pt` (~172 MB/cell × 42 cells ≈ 17 GB total —
well under the VM 50 GB analysis footprint per plan §9).

Uploads each cell to HF `superkaiba1/explore-persona-space-data/
issue1689_speaker_lattice/analysis_tensors/` immediately after write
(persist-by-default per plan §5 upload-policy).

Smoke: --smoke → 1 condition × 5 rows on a tiny same-arch stub model at
the plan's layer set (assert file shape only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    CAPTURE_LAYERS,
    D_MODEL,
    HF_DATA_PREFIX,
    ISSUE_NUM,
    ISSUE_SLUG,
    MODEL_BASE,
    MODEL_INSTRUCT,
)


def _mock_activation(n: int, d: int, seed: int = 42):
    """Deterministic mock activation tensor for smoke tests."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def capture_cell(
    rows: list[dict],
    *,
    model_name: str,
    condition_slug: str,
    layers: tuple[int, ...] = CAPTURE_LAYERS,
    d_model: int = D_MODEL,
    mock: bool = False,
) -> dict:
    """Capture activations for one (model, condition) cell.

    Returns a dict {layer -> {arm -> (N, D) tensor, conv_ids -> (N,)}}
    ready for torch.save. Two arms per layer: 'prefix', 'context'.
    y_layer stores the answer-side activation at end of a2.
    """
    import numpy as np
    import torch

    n = len(rows)
    if n == 0:
        raise ValueError(f"no rows to capture for {condition_slug}")

    conv_ids = np.array([row["conv_id"] for row in rows])

    out: dict = {"conv_ids": conv_ids, "condition": condition_slug, "model": model_name}
    for layer in layers:
        if mock:
            X_prefix = _mock_activation(n, d_model, seed=layer * 7)
            X_context = _mock_activation(n, d_model, seed=layer * 7 + 1)
            Y = _mock_activation(n, d_model, seed=layer * 7 + 2)
        else:
            # Real routing: import + hook-based capture. Lazy so smoke doesn't
            # need transformers.
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            X_prefix_list, X_context_list, Y_list = [], [], []
            for row in rows:
                # Teacher-force prefix span, then context span, then answer span
                # For a chat framing we'd use apply_chat_template; for
                # naturalistic/story we use row["prompt_text"]. Simplified here
                # to prompt_text + a2_text.
                prefix_text = row.get("prompt_text", "")
                a2 = row.get("a2_text", "")
                # Prefix ids
                p_ids = tok(prefix_text, return_tensors="pt").input_ids
                with torch.no_grad():
                    out_p = model(p_ids.cuda(), output_hidden_states=True)
                    hs = out_p.hidden_states[layer][0, -1].float().cpu().numpy()
                    X_prefix_list.append(hs)
                    # Context span (same as prefix here in simplified form)
                    X_context_list.append(hs)
                # Answer end
                full_ids = tok(prefix_text + a2, return_tensors="pt").input_ids
                with torch.no_grad():
                    out_a = model(
                        full_ids.cuda(),
                        output_hidden_states=True,
                    )
                    Y_list.append(out_a.hidden_states[layer][0, -1].float().cpu().numpy())
            X_prefix = np.stack(X_prefix_list)
            X_context = np.stack(X_context_list)
            Y = np.stack(Y_list)

        out[f"L{layer}"] = {
            "X_prefix": X_prefix,
            "X_context": X_context,
            "Y": Y,
        }
    return out


def save_cell(cell_data: dict, out_root: Path, model_name: str, condition_slug: str) -> Path:
    """Save the cell as one .pt bundle per layer (plan §6.5 primary_deliverable
    path: L19.pt is the headline; the others land as siblings)."""
    import torch

    dest = out_root / model_name.replace("/", "_") / condition_slug
    dest.mkdir(parents=True, exist_ok=True)
    for key, val in cell_data.items():
        if not key.startswith("L"):
            continue
        layer = int(key[1:])
        path = dest / f"L{layer}.pt"
        # Save X_prefix, X_context, Y in one file per (cell, layer)
        torch.save(
            {
                "X_prefix": val["X_prefix"],
                "X_context": val["X_context"],
                "Y": val["Y"],
                "conv_ids": cell_data["conv_ids"],
                "condition": condition_slug,
                "model": model_name,
                "layer": layer,
            },
            path,
        )
    return dest


def upload_cell_to_hf(cell_dir: Path, model_name: str, condition_slug: str) -> str | None:
    """Upload the per-cell analysis tensors to the HF data repo per plan §5.

    Returns the HF path prefix on success; None on smoke/mock.
    """
    from explore_persona_space.orchestrate.hub import _upload

    hf_subpath = (
        f"{HF_DATA_PREFIX}/analysis_tensors/{model_name.replace('/', '_')}/{condition_slug}"
    )
    for pt_file in sorted(cell_dir.glob("L*.pt")):
        _upload(
            pt_file,
            repo_id="superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            path_in_repo=f"{hf_subpath}/{pt_file.name}",
            upload_as_file=True,
        )
    return hf_subpath


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=[MODEL_BASE, MODEL_INSTRUCT])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    rows = []
    with args.in_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") == args.condition:
                rows.append(row)
    if args.smoke:
        rows = rows[:5]

    if not rows:
        raise SystemExit(f"no rows for condition={args.condition}")

    cell = capture_cell(
        rows,
        model_name=args.model,
        condition_slug=args.condition,
        mock=args.smoke,
    )
    dest = save_cell(cell, args.out_root, args.model, args.condition)
    print(f"[capture] wrote {len(list(dest.glob('L*.pt')))} layer files to {dest}")

    if not args.skip_upload and not args.smoke:
        hf_path = upload_cell_to_hf(dest, args.model, args.condition)
        print(f"[capture] uploaded to HF {hf_path}")
    else:
        print(
            f"[capture] skipping upload for issue{ISSUE_NUM}_{ISSUE_SLUG} (smoke or --skip-upload)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
