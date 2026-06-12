"""Issue #621 CPU smoke driver — exercises every CPU-verifiable code path.

Builds a TINY random Qwen2 model (26 layers, hidden 16, FULL 152064 vocab so
the real marker/im_end ids are in range; RMSNorm γ perturbed NON-uniformly
— a γ=ones model false-PASSes a∘γ bugs) + the REAL Qwen-2.5-7B-Instruct
tokenizer, then drives the PRODUCTION entrypoints against fixture roots:

  a-init        _make_initial_adapter_snapshot_callback.on_train_begin on a
                rank-1 PEFT wrap → adapter_init exists, B exactly zero.
  shift         extract_per_context_shift (tiny base + rank-1 trained twin,
                2 questions) → four-float slot stats + per-question arrays.
  bank-capture  scripts/issue621_extract_context_bank.py --step capture on
                a 21-context × 2-probe responses.json fixture (the REAL CLI;
                o_proj/down_proj pre-hooks, 3 positions, manifest assert).
  analyze       scripts/issue621_analyze.py run on 6 synthetic rank-1 cells
                (3 arms × 2 seeds) + the bank-capture bundle + synthetic
                eval shift JSONs + a fixture unembedding (the REAL CLI).
  figures       scripts/issue621_figures.py on the produced analysis.json.

GPU-bound paths (train_lora forward, vLLM generate/emission) are covered by
the dispatcher dry-runs + signature smokes (carve-out); everything here
runs the real code on CPU. Exit 0 = all smoke phases pass.

CLI:
    uv run python scripts/i621_cpu_smoke.py --work-dir /tmp/i621_cpu_smoke
"""

# ruff: noqa: RUF002, RUF003  # math notation

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_621 import (
    MARKER_ID,
    PLACEMENT_ARMS,
)
from explore_persona_space.experiments.issue_621.persona_registry import load_persona_bank

log = logging.getLogger("issue_621.cpu_smoke")

SCRIPTS = Path(__file__).resolve().parent
TINY_LAYERS = 26  # > BAND_LAYERS max (24) and EXTRACTION_LAYER (20)
TINY_HIDDEN = 16
TINY_DFF = 32
SMOKE_CELLS = [
    ("read", "florist", 42),
    ("read", "florist", 137),
    ("write", "florist", 42),
    ("write", "florist", 137),
    ("bridge", "florist", 42),
    ("bridge", "florist", 137),
]


def _run_cli(args: list[str], label: str) -> None:
    """Run a production CLI as a subprocess; fail loud with the log tail."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:] + "\n")
        raise SystemExit(f"CPU smoke phase {label!r} FAILED (rc={proc.returncode})")
    log.info("CLI smoke %s PASS", label)


def build_tiny_model(work: Path) -> Path:
    """Random tiny Qwen2 + the REAL tokenizer; γ perturbed non-uniformly."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    tiny_dir = work / "tiny_model"
    if (tiny_dir / "config.json").is_file():
        log.info("tiny model exists at %s", tiny_dir)
        return tiny_dir
    cfg = Qwen2Config(
        vocab_size=152064,
        hidden_size=TINY_HIDDEN,
        intermediate_size=TINY_DFF,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    torch.manual_seed(621)
    model = AutoModelForCausalLM.from_config(cfg)
    # Perturb RMSNorm γ non-uniformly (ones-init γ false-PASSes a∘γ bugs).
    with torch.no_grad():
        for layer in model.model.layers:
            layer.input_layernorm.weight.add_(0.1 * torch.randn(TINY_HIDDEN))
            layer.post_attention_layernorm.weight.add_(0.1 * torch.randn(TINY_HIDDEN))
    tiny_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tiny_dir)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    tok.save_pretrained(tiny_dir)
    assert tok.encode(" ※", add_special_tokens=False) == [MARKER_ID]
    log.info("tiny model + real tokenizer saved to %s", tiny_dir)
    return tiny_dir


def _make_rank1_adapter(tiny_dir: Path, arm: str, seed: int, out_dir: Path) -> None:
    """PEFT rank-1 wrap per arm; save the zero-B init + a perturbed 'final'."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32)
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=1,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=list(PLACEMENT_ARMS[arm]),
        use_rslora=True,
    )
    pm = get_peft_model(model, peft_cfg)
    init_dir = out_dir / "adapter_init"
    init_dir.mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(init_dir))
    # adapter_init/<default>/... vs flat: PEFT saves flat for single adapter.
    if not (init_dir / "adapter_model.safetensors").is_file():
        raise SystemExit(f"init save shape unexpected under {init_dir}")
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if "lora_A" in name:
                p.add_(0.05 * torch.randn_like(p))
            elif "lora_B" in name:
                p.copy_(0.1 * torch.randn_like(p))
    pm.save_pretrained(str(out_dir))
    assert (out_dir / "adapter_model.safetensors").is_file()


def smoke_a_init(work: Path, tiny_dir: Path) -> None:
    """Unit smoke of the sft.py snapshot callback on a real PEFT model."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from explore_persona_space.train.sft import _make_initial_adapter_snapshot_callback

    out = work / "a_init_smoke"
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32)
    pm = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=1,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            use_rslora=True,
        ),
    )
    cb = _make_initial_adapter_snapshot_callback(out)
    cb.on_train_begin(None, None, None, model=pm)
    snap = out / "adapter_init" / "adapter_model.safetensors"
    assert snap.is_file(), snap
    sd = load_file(str(snap))
    b_keys = [k for k in sd if "lora_B" in k]
    a_keys = [k for k in sd if "lora_A" in k]
    assert b_keys and a_keys, (len(a_keys), len(b_keys))
    assert all(float(sd[k].abs().max()) == 0.0 for k in b_keys), "B_init not zero"
    assert any(float(sd[k].abs().max()) > 0.0 for k in a_keys), "A_init all-zero?"
    # model=None / non-PEFT must fail LOUD.
    for bad in (None, object()):
        try:
            cb.on_train_begin(None, None, None, model=bad)
        except RuntimeError:
            pass
        else:
            raise SystemExit(f"snapshot callback accepted bad model={bad!r}")
    log.info("a-init callback smoke PASS (%d A tensors, %d B tensors)", len(a_keys), len(b_keys))


def smoke_shift_extract(work: Path, tiny_dir: Path) -> None:
    """extract_per_context_shift on tiny base + rank-1 trained twin (CPU)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.experiments.issue_621.shift_extract import (
        extract_per_context_shift,
    )
    from explore_persona_space.personas import EVAL_QUESTIONS

    adapter_dir = work / "shift_adapter"
    _make_rank1_adapter(tiny_dir, "read", 42, adapter_dir)
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    base = AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32).eval()
    trained = AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32)
    trained = PeftModel.from_pretrained(trained, str(adapter_dir)).eval()

    bank = load_persona_bank()
    qs = list(EVAL_QUESTIONS[:2])
    cs = extract_per_context_shift(
        base_model=base,
        trained_model=trained,
        tokenizer=tok,
        persona="florist",
        persona_prompt=bank["florist"],
        eval_questions=qs,
        r_responses={q: "A short fixture response." for q in qs},
        device="cpu",
    )
    assert cs.n_prompts == 2
    assert cs.shift_vector.shape == (TINY_HIDDEN,), tuple(cs.shift_vector.shape)
    assert len(cs.per_question_delta_logp) == 2
    assert len(cs.per_question_delta_margin) == 2
    for side in (cs.marker_slot_stats_trained, cs.marker_slot_stats_base):
        for v in (side.logp_marker, side.z_marker, side.z_eos, side.logZ):
            assert np.isfinite(v), side
        # Bookkeeping identity: logp = z_marker − logZ (fp tolerance).
        assert abs(side.logp_marker - (side.z_marker - side.logZ)) < 1e-3
    log.info(
        "shift_extract smoke PASS (Δlogp=%.4f, slot=%.1f)",
        cs.delta_logp_marker,
        cs.slot_index_mean,
    )


def _bank_fixture_responses(work: Path, tiny_dir: Path) -> Path:
    """responses.json fixture: ALL 21 assembled contexts × 2 probes."""
    sys.path.insert(0, str(SCRIPTS))
    from issue621_extract_context_bank import assemble_contexts

    contexts = assemble_contexts()
    probes = ["What is your favorite season?", "Describe a useful tool."]
    payload = {
        "schema_version": "issue_621_bank_responses_v1",
        "model": str(tiny_dir),
        "max_new_tokens": 16,
        "n_probes": len(probes),
        "probes": probes,
        "probes_sha256": hashlib.sha256("\x00".join(probes).encode()).hexdigest(),
        "truncation_rate": 0.0,
        "contexts": {
            name: {
                "system_prompt": ctx["system_prompt"],
                "groups": ctx["groups"],
                "prompt_sha256": ctx["prompt_sha256"],
                "responses": [
                    f"Fixture response one for {name}.",
                    f"A different fixture response for {name}, slightly longer.",
                ],
                "truncated": [False, False],
            }
            for name, ctx in contexts.items()
        },
        "git_commit": "smoke",
        "timestamp_utc": "1970-01-01T00:00:00+00:00",
    }
    out = work / "bank_fixture_responses.json"
    out.write_text(json.dumps(payload, indent=1))
    log.info("bank fixture: %d contexts x %d probes", len(contexts), len(probes))
    return out


def smoke_bank(work: Path, tiny_dir: Path) -> Path:
    """generate --dry-run (real loader) + capture CLI on the fixture."""
    bank_dir = work / "bank"
    _run_cli(
        [
            "uv",
            "run",
            "python",
            str(SCRIPTS / "issue621_extract_context_bank.py"),
            "--step",
            "generate",
            "--dry-run",
            "--probes",
            "2",
            "--model",
            str(tiny_dir),
            "--context-names",
            "florist,chef",
            "--out-dir",
            str(work / "bank_dry"),
        ],
        "bank generate --dry-run",
    )
    responses = _bank_fixture_responses(work, tiny_dir)
    _run_cli(
        [
            "uv",
            "run",
            "python",
            str(SCRIPTS / "issue621_extract_context_bank.py"),
            "--step",
            "capture",
            "--model",
            str(tiny_dir),
            "--dtype",
            "float32",
            "--responses-json",
            str(responses),
            "--out-dir",
            str(bank_dir),
        ],
        "bank capture (tiny CPU)",
    )
    manifest = json.loads((bank_dir / "manifest.json").read_text())
    assert "o_in" in manifest["meta"]["taps"] and "down_in" in manifest["meta"]["taps"]
    import torch

    cents = torch.load(bank_dir / "centroids.pt", weights_only=True)["centroids"]
    t = cents["down_in"]["end_of_response"]["florist"]
    assert tuple(t.shape) == (TINY_LAYERS, TINY_DFF), tuple(t.shape)
    t2 = cents["o_in"]["response_mean"]["chef"]
    assert tuple(t2.shape) == (TINY_LAYERS, TINY_HIDDEN), tuple(t2.shape)
    log.info("bank capture smoke PASS (5 taps x 3 positions, manifest assert exercised)")
    return bank_dir


def smoke_analyze_and_figures(work: Path, tiny_dir: Path, bank_dir: Path) -> None:
    """Synthetic 6-cell fixture → REAL analyze + figures CLIs."""
    import torch

    from explore_persona_space.experiments.issue_621 import cell_slug

    rng = np.random.default_rng(621)
    adapters_root = work / "adapters"
    eval_root = work / "eval"
    cells_root = work / "cells_root"
    (cells_root / "sweep").mkdir(parents=True, exist_ok=True)
    (cells_root / "anchor_smoke").mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    panel = json.loads(
        (bank_dir / "manifest.json").read_text(),
    )["contexts"]
    panel_names = [n for n in panel if n not in ("programmer", "chef")]  # 19-panel
    assert len(panel_names) == 19, len(panel_names)

    for arm, source, seed in SMOKE_CELLS:
        slug = cell_slug(arm, source, seed)
        cell_dir = adapters_root / slug
        _make_rank1_adapter(tiny_dir, arm, seed, cell_dir)
        # Train-side metadata JSON (band metadata joined by the analyzer).
        sub = "anchor_smoke" if (arm, source, seed) == ("read", "florist", 42) else "sweep"
        (cells_root / sub / f"{slug}.json").write_text(
            json.dumps(
                {
                    "cell_slug": slug,
                    "arm": arm,
                    "source": source,
                    "seed": seed,
                    "band_stop_fired": bool(seed == 42),
                    "band_stop_step": 30 if seed == 42 else None,
                    "final_source_delta_nats": 7.5 if seed == 42 else 3.1,
                    "hf_subfolder": f"adapters/issue_621/{slug}",
                    "output_dir": str(cell_dir),
                    "wandb_run_name": f"issue621_{slug}",
                }
            )
        )
        # Synthetic eval shift JSON + .pt.
        contexts = {}
        for p in panel_names:
            zt, zb = rng.normal(0, 2), rng.normal(-3, 2)
            et, eb = rng.normal(8, 1), rng.normal(8, 1)
            contexts[p] = {
                "n_prompts": 5,
                "delta_logp_marker": float(rng.normal(2, 3)),
                "delta_logit_marker": float(rng.normal(2, 3)),
                "emission_argmax_trained": float(rng.uniform(0, 0.5)),
                "emission_argmax_base": 0.0,
                "per_question_delta_logp": [float(x) for x in rng.normal(2, 1, size=5)],
                "per_question_delta_margin": [float(x) for x in rng.normal(1, 1, size=5)],
                "marker_slot_stats": {
                    "trained": {
                        "logp_marker": float(rng.normal(-8, 2)),
                        "z_marker": float(zt),
                        "z_eos": float(et),
                        "logZ": float(zt + 10),
                    },
                    "base": {
                        "logp_marker": float(rng.normal(-18, 2)),
                        "z_marker": float(zb),
                        "z_eos": float(eb),
                        "logZ": float(zb + 20),
                    },
                    "slot_index": 60.0,
                },
            }
        (eval_root / f"{slug}__shift.json").write_text(
            json.dumps(
                {
                    "schema_version": "issue_621_shift_v1",
                    "cell_slug": slug,
                    "arm": arm,
                    "source": source,
                    "seed": seed,
                    "eval_panel": panel_names,
                    "eval_questions": ["q1", "q2"],
                    "contexts": contexts,
                    "shift_matrix_path": f"{slug}__shift.pt",
                }
            )
        )
        torch.save(
            torch.from_numpy(
                np.asarray(rng.normal(size=(len(panel_names), TINY_HIDDEN)), dtype=np.float32)
            ),
            eval_root / f"{slug}__shift.pt",
        )

    # Unembedding fixture (tiny dims).
    wu_path = work / "unembedding_rows.pt"
    torch.save(
        {
            "W_U_marker": torch.randn(TINY_HIDDEN),
            "W_U_eos": torch.randn(TINY_HIDDEN),
            "W_U_null_norm_matched": torch.randn(50, TINY_HIDDEN),
            "W_U_null_random": torch.randn(50, TINY_HIDDEN),
        },
        wu_path,
    )

    out_dir = work / "analysis"
    _run_cli(
        [
            "uv",
            "run",
            "python",
            str(SCRIPTS / "issue621_analyze.py"),
            "run",
            "--adapters-root",
            str(adapters_root),
            "--bank",
            str(bank_dir),
            "--eval-root",
            str(eval_root),
            "--cells-root",
            str(cells_root),
            "--unembedding",
            str(wu_path),
            "--out",
            str(out_dir),
        ],
        "analyze run (6 synthetic cells, tiny bank)",
    )
    payload = json.loads((out_dir / "analysis.json").read_text())
    assert len(payload["cells"]) == 6, len(payload["cells"])
    c = payload["cells"]["r1_write__florist__seed42"]
    assert c["write_identity"], "write arm produced no W_U reads"
    assert c["h4"]["end_of_response"]["write"]["primary_excl_trained_negs"]
    assert payload["summary"]["cross_seed"], "cross-seed block empty"

    fig_dir = work / "figures"
    _run_cli(
        [
            "uv",
            "run",
            "python",
            str(SCRIPTS / "issue621_figures.py"),
            "--analysis",
            str(out_dir / "analysis.json"),
            "--out-dir",
            str(fig_dir),
        ],
        "figures",
    )
    pngs = sorted(fig_dir.glob("*.png"))
    assert len(pngs) >= 4, [p.name for p in pngs]
    log.info(
        "analyze + figures smoke PASS (%d cells, %d figures)", len(payload["cells"]), len(pngs)
    )


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-dir", default="/tmp/i621_cpu_smoke")
    args = ap.parse_args(argv)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    tiny_dir = build_tiny_model(work)
    smoke_a_init(work, tiny_dir)
    smoke_shift_extract(work, tiny_dir)
    bank_dir = smoke_bank(work, tiny_dir)
    smoke_analyze_and_figures(work, tiny_dir, bank_dir)
    log.info("ALL CPU smoke phases PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
