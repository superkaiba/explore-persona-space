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
                (3 arms × 2 seeds, each with a 3-step checkpoint ladder for
                the duty-10 a(t) read) + the bank-capture bundle + synthetic
                eval shift JSONs + a fixture unembedding (the REAL CLI);
                asserts the duty-10 trajectory + duty-12 split-half /
                paired-diff outputs (round 3).
  a-init-gauge  analyze_cell on a CONTROLLED write-arm cell whose every
                residual-output slot has W_U[※]·b < 0 — proves the H2
                a_init reads run in the UNFLIPPED pair gauge (frozen-A
                layers read rel_delta_a == 0, eps-rotated layers read
                cos(Δâ, v̂_src) ≈ +1; round 4).
  figures       scripts/issue621_figures.py on the produced analysis.json.
  fetch         cmd_fetch_artifacts against a LOCAL STUB HUB shaped like the
                production repos (adapters / bank / shifts / emission /
                train_meta / trajectories), then the REAL analyze ``run`` on
                the FETCHED layout — the clean-VM off-pod Phase A path
                (round-2, concern analysis-fetch-missing-shifts).

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


def _make_rank1_adapter(
    tiny_dir: Path,
    arm: str,
    seed: int,
    out_dir: Path,
    checkpoint_steps: tuple[int, ...] = (),
    final_extra_perturb: bool = True,
) -> None:
    """PEFT rank-1 wrap per arm; save the zero-B init + a perturbed 'final'.

    With ``checkpoint_steps``, also saves a progressively-perturbed
    ``checkpoint-<N>/`` ladder (the duty-10 a(t) trajectory fixture; same
    flat PEFT layout Trainer ``save_only_model`` checkpoints carry).
    ``final_extra_perturb=False`` makes the final state identical to the
    last checkpoint — exercising the analyzer's terminal-point dedup
    (band-stop landing exactly on a save step).
    """
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

    def _perturb() -> None:
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_A" in name:
                    p.add_(0.05 * torch.randn_like(p))
                elif "lora_B" in name:
                    p.add_(0.1 * torch.randn_like(p))

    for step in checkpoint_steps:
        _perturb()
        ck_dir = out_dir / f"checkpoint-{step}"
        ck_dir.mkdir(parents=True, exist_ok=True)
        pm.save_pretrained(str(ck_dir))
        assert (ck_dir / "adapter_model.safetensors").is_file()
    if final_extra_perturb or not checkpoint_steps:
        _perturb()
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
    # Round-2 (concern duty12-split-half-inputs-missing): per-question
    # PER-SIDE four-float contract (+ derived margin) — present, aligned,
    # and internally consistent with the per-question deltas.
    pq_t = cs.per_question_slot_stats_trained
    pq_b = cs.per_question_slot_stats_base
    expected_keys = {"logp_marker", "z_marker", "z_eos", "logZ", "margin"}
    for side in (pq_t, pq_b):
        assert set(side) == expected_keys, sorted(side)
        assert all(len(v) == 2 for v in side.values()), {k: len(v) for k, v in side.items()}
        assert all(np.isfinite(v) for vals in side.values() for v in vals)
    for i in range(2):
        for side in (pq_t, pq_b):
            assert abs(side["logp_marker"][i] - (side["z_marker"][i] - side["logZ"][i])) < 1e-3
            assert abs(side["margin"][i] - (side["z_marker"][i] - side["z_eos"][i])) < 1e-6
        assert (
            abs(cs.per_question_delta_logp[i] - (pq_t["logp_marker"][i] - pq_b["logp_marker"][i]))
            < 1e-6
        )
        assert abs(cs.per_question_delta_margin[i] - (pq_t["margin"][i] - pq_b["margin"][i])) < 1e-6
    log.info(
        "shift_extract smoke PASS (Δlogp=%.4f, slot=%.1f, per-q per-side stats verified)",
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


def smoke_analyze_and_figures(work: Path, tiny_dir: Path, bank_dir: Path) -> dict[str, Path]:
    """Synthetic 6-cell fixture → REAL analyze + figures CLIs.

    Returns the fixture roots so the fetch-artifacts smoke can republish
    them through a local stub hub and re-run analyze on the FETCHED layout.
    """
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
        # Duty-10 fixture: a 3-step checkpoint ladder per cell. The bridge
        # seed-42 cell keeps final == checkpoint-30 (band-stop exactly on a
        # save step) to exercise the analyzer's terminal-point dedup.
        _make_rank1_adapter(
            tiny_dir,
            arm,
            seed,
            cell_dir,
            checkpoint_steps=(10, 20, 30),
            final_extra_perturb=(arm, source, seed) != ("bridge", "florist", 42),
        )
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
            # Round-2 schema: per-question PER-SIDE four-float contract
            # (+ derived margin), consistent with the per-question deltas.
            pq_side = {}
            for side_name, z_mu, eos_mu in (("trained", zt, et), ("base", zb, eb)):
                z = rng.normal(z_mu, 1, size=5)
                eos = rng.normal(eos_mu, 1, size=5)
                logZ = z + rng.uniform(8, 22, size=5)
                pq_side[side_name] = {
                    "logp_marker": [float(v) for v in (z - logZ)],
                    "z_marker": [float(v) for v in z],
                    "z_eos": [float(v) for v in eos],
                    "logZ": [float(v) for v in logZ],
                    "margin": [float(v) for v in (z - eos)],
                }
            pq_delta_logp = [
                t - b
                for t, b in zip(
                    pq_side["trained"]["logp_marker"], pq_side["base"]["logp_marker"], strict=True
                )
            ]
            pq_delta_margin = [
                t - b
                for t, b in zip(
                    pq_side["trained"]["margin"], pq_side["base"]["margin"], strict=True
                )
            ]
            contexts[p] = {
                "n_prompts": 5,
                "delta_logp_marker": float(rng.normal(2, 3)),
                "delta_logit_marker": float(rng.normal(2, 3)),
                "emission_argmax_trained": float(rng.uniform(0, 0.5)),
                "emission_argmax_base": 0.0,
                "per_question_delta_logp": pq_delta_logp,
                "per_question_delta_margin": pq_delta_margin,
                "per_question_slot_stats": pq_side,
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
            # 6 synthetic cells < the 30-cell enumerated grid — the smoke is
            # the deliberate-subset case the escape hatch exists for.
            "--allow-partial",
        ],
        "analyze run (6 synthetic cells, tiny bank)",
    )
    payload = json.loads((out_dir / "analysis.json").read_text())
    assert len(payload["cells"]) == 6, len(payload["cells"])
    # Round 4 (concern h2-a-init-reads-after-sign-flip): the gauge
    # convention for a_init-referenced reads is recorded in meta + per cell.
    assert payload["meta"]["a_init_gauge"] == "unflipped-pair", payload["meta"].get("a_init_gauge")
    assert all(c["a_init_gauge"] == "unflipped-pair" for c in payload["cells"].values())
    c = payload["cells"]["r1_write__florist__seed42"]
    assert c["write_identity"], "write arm produced no W_U reads"
    assert c["h4"]["end_of_response"]["write"]["primary_excl_trained_negs"]
    assert payload["summary"]["cross_seed"], "cross-seed block empty"

    # Duty 10 (round 3): a(t) rotation trajectory rides the checkpoint ladder.
    traj = payload["cells"]["r1_read__florist__seed42"]["a_rotation_trajectory"]
    assert traj["n_checkpoints"] == 3, traj["n_checkpoints"]
    assert traj["checkpoint_steps"] == [10, 20, 30], traj["checkpoint_steps"]
    assert len(traj["points"]) == 4, traj["points"]  # 3 ckpts + distinct final
    assert traj["points"][-1]["is_final"] and traj["points"][-1]["step"] == 30
    assert all(
        np.isfinite(pt["band_mean_abs_cos_a_init"]) and np.isfinite(pt["band_mean_rel_delta_a"])
        for pt in traj["points"]
    )
    assert traj["rotation_still_growing_at_stop"] is not None
    # Terminal-point dedup: the bridge seed-42 final state == checkpoint-30.
    traj_b = payload["cells"]["r1_bridge__florist__seed42"]["a_rotation_trajectory"]
    assert traj_b["n_checkpoints"] == 3 and len(traj_b["points"]) == 3, traj_b
    assert traj_b["final_terminal_point_deduped"] is True
    assert (out_dir / "a_trajectory__r1_read__florist__seed42.json").is_file()
    assert payload["summary"]["a_rotation"]["r1_read__florist__seed42"]["n_checkpoints"] == 3
    assert payload["summary"]["cells_missing_ladder"] == []

    # Duty 12 (round 3): split-half base check + paired comparator diffs.
    sh = payload["cells"]["r1_read__florist__seed42"]["split_half_base_check"]
    assert set(sh["splits"]) == {"prior_even_dv_odd", "prior_odd_dv_even"}, sh["splits"].keys()
    assert sh["n_questions"] == 5 and sh["halves"] == {"even": 3, "odd": 2}
    for block in sh["splits"].values():
        for v in block.values():
            assert np.isfinite(v), block
    assert (
        payload["cells"]["r1_write__florist__seed42"]["split_half_base_check"]["firing_mode"]
        == "write"
    )
    pcd = payload["summary"]["h4_paired_comparator_diffs"]
    expected_cmps = {"base_prior", "geometry_centered", "context_norm", "a_init_firing"}
    assert set(pcd["read"]) == expected_cmps, sorted(pcd["read"])
    row = pcd["read"]["base_prior"]
    assert row["n_cells"] == 2 and row["n_sources"] == 1, row  # 1 source x 2 seeds
    assert np.isfinite(row["mean_diff_over_sources"])
    assert payload["summary"]["split_half_base_check_by_arm"]["read"]

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
    return {
        "adapters_root": adapters_root,
        "eval_root": eval_root,
        "cells_root": cells_root,
        "wu_path": wu_path,
        "bank_dir": bank_dir,
    }


def smoke_a_init_gauge(work: Path, bank_dir: Path, fixture_roots: dict[str, Path]) -> None:
    """Round-4 targeted check (concern h2-a-init-reads-after-sign-flip).

    Builds a CONTROLLED write-arm cell whose EVERY residual-output slot has
    W_U[※]·b < 0 (the all-flipped worst case for ``_sign_fix_write``), with
    a_t frozen at a_init on even layers and rotated by exactly 0.05·v̂_src
    on odd layers, then drives the REAL ``analyze_cell``. Under the round-3
    bug (a_init reads compared the FLIPPED a_t against the unflipped
    a_init) every slot reads rel_delta_a ≈ 2 and cos(Δâ, v̂_c)
    sign-corrupted; under the fix the frozen layers read exactly 0 and the
    rotated layers read rel ≈ 0.05 / cos ≈ +1.
    """
    import shutil

    from safetensors.numpy import save_file

    sys.path.insert(0, str(SCRIPTS))
    import issue621_analyze as iaz

    rng = np.random.default_rng(4621)
    bank = iaz.load_bank(bank_dir)
    wu = iaz.load_unembedding(fixture_roots["wu_path"])

    slug = "r1_write__florist__seed7"
    cell_dir = work / "gauge_cell" / slug
    if cell_dir.is_dir():
        shutil.rmtree(cell_dir)
    init_dir = cell_dir / "adapter_init"
    init_dir.mkdir(parents=True, exist_ok=True)
    # Reuse a real PEFT write-arm adapter_config (the gauge assert reads it).
    cfg_src = fixture_roots["adapters_root"] / "r1_write__florist__seed42" / "adapter_config.json"
    shutil.copy2(cfg_src, cell_dir / "adapter_config.json")
    shutil.copy2(cfg_src, init_dir / "adapter_config.json")

    eps = 0.05
    dims = {"o_proj": ("self_attn", TINY_HIDDEN), "down_proj": ("mlp", TINY_DFF)}
    sd_init: dict[str, np.ndarray] = {}
    sd_final: dict[str, np.ndarray] = {}
    for module, (parent, d_in) in dims.items():
        tap = iaz.MODULE_INPUT_TAP[module]
        v_src = bank["centroids"][tap][iaz.PRIMARY_POSITION]["florist"]  # (L, d_in)
        for li in range(TINY_LAYERS):
            a0 = rng.normal(size=d_in).astype(np.float32)
            a0 /= np.linalg.norm(a0)  # unit init → rotated rel == eps exactly
            b = rng.normal(size=TINY_HIDDEN).astype(np.float32)
            if float(wu["marker"] @ b) > 0:
                b = -b
            # Fixture-construction assert: every slot sits in the flip branch.
            assert float(wu["marker"] @ b) < 0, (module, li)
            if li % 2 == 0:
                a_t = a0.copy()  # frozen-A stub: Δa == 0 exactly
            else:
                v = np.asarray(v_src[li], dtype=np.float32)
                v_norm = float(np.linalg.norm(v))
                assert v_norm > 0, (module, li, "zero-norm bank centroid")
                a_t = (a0 + np.float32(eps) * (v / v_norm)).astype(np.float32)
            key = f"base_model.model.model.layers.{li}.{parent}.{module}"
            sd_init[f"{key}.lora_A.weight"] = a0[None, :]
            sd_init[f"{key}.lora_B.weight"] = np.zeros((TINY_HIDDEN, 1), dtype=np.float32)
            sd_final[f"{key}.lora_A.weight"] = a_t[None, :]
            sd_final[f"{key}.lora_B.weight"] = b[:, None]
    save_file(sd_init, str(init_dir / "adapter_model.safetensors"))
    save_file(sd_final, str(cell_dir / "adapter_model.safetensors"))

    # Eval payload: reuse the synthetic write-cell fixture, re-keyed to the
    # controlled slug (load_eval_cells joins exactly these two extra keys).
    eval_root = fixture_roots["eval_root"]
    payload = json.loads((eval_root / "r1_write__florist__seed42__shift.json").read_text())
    payload["cell_slug"] = slug
    payload["seed"] = 7
    payload["_train_meta"] = {"band_stop_fired": True, "band_stop_step": 30}
    payload["_shift_pt_path"] = str(eval_root / "r1_write__florist__seed42__shift.pt")

    res = iaz.analyze_cell(slug=slug, adapter_dir=cell_dir, bank=bank, eval_payload=payload, wu=wu)
    assert res["a_init_gauge"] == "unflipped-pair", res.get("a_init_gauge")
    for module in ("o_proj", "down_proj"):
        block = res["a_init"][module]
        rels = block["rel_delta_a_per_layer"]
        coss = block["cos_delta_a_vs_vc_per_layer"]
        cos_ai = block["cos_a_init_per_layer"]
        assert len(rels) == TINY_LAYERS, (module, len(rels))
        for li in range(TINY_LAYERS):
            if li % 2 == 0:  # frozen-A: Δa == 0 exactly (bug ⇒ rel ≈ 2)
                assert rels[li] == 0.0, (module, li, rels[li])
                assert coss[li] == 0.0, (module, li, coss[li])  # zero-vec guard
                assert abs(cos_ai[li] - 1.0) < 1e-5, (module, li, cos_ai[li])
            else:  # rotated by exactly eps·v̂_src (bug ⇒ cos sign-corrupted)
                assert abs(rels[li] - eps) < 1e-4, (module, li, rels[li])
                assert coss[li] > 0.99, (module, li, coss[li])
        # Band means: bug value ≈ 2.0 on the all-flipped cell; fix ≈ eps/2.
        assert block["band_mean_rel_delta_a"] < 0.5, block["band_mean_rel_delta_a"]
        assert block["band_mean_cos_a_init"] > 0.95, block["band_mean_cos_a_init"]
    log.info(
        "a_init gauge smoke PASS (all-flipped write cell: frozen rel==0, rotated cos(Δâ,v̂)≈+1)"
    )


def smoke_fetch_artifacts(work: Path, fixture_roots: dict[str, Path]) -> None:
    """cmd_fetch_artifacts against a LOCAL STUB HUB, then analyze the fetch.

    Round-2 (concern analysis-fetch-missing-shifts): the clean-VM Phase A
    path is ``fetch-artifacts && build-unembedding && run``. This smoke
    republishes the analyze fixtures through a stub hub tree shaped EXACTLY
    like the production repos (model repo: ``adapters/issue_621/<slug>/``;
    data repo: ``analysis_tensors/{bank,shifts}``, ``eval/``,
    ``train_meta/``, ``trajectories/``), monkeypatches the two Hub calls
    ``cmd_fetch_artifacts`` makes (``list_repo_files_complete`` +
    ``hf_hub_download``), fetches into a FRESH root, asserts the local
    layout ``run`` requires, and re-runs the REAL analyze ``run`` on the
    fetched layout — proving the off-pod path works end-to-end without the
    instance.
    """
    import shutil
    from unittest import mock

    import explore_persona_space.orchestrate.hub as eps_hub
    from explore_persona_space.experiments.issue_621 import (
        HF_ADAPTER_PATH_PREFIX,
        HF_ANALYSIS_TENSORS_PREFIX,
        HF_BUCKET,
    )

    sys.path.insert(0, str(SCRIPTS))
    import issue621_analyze as iaz

    adapters_root = fixture_roots["adapters_root"]
    eval_root = fixture_roots["eval_root"]
    cells_root = fixture_roots["cells_root"]
    bank_dir = fixture_roots["bank_dir"]

    # ── Build the stub hub tree from the fixtures ────────────────────────
    hub_root = work / "stub_hub"
    if hub_root.is_dir():
        shutil.rmtree(hub_root)
    model_root = hub_root / "model"
    data_root = hub_root / "data"
    # Model repo: adapter + adapter_init + checkpoint-*/ ladder under
    # adapters/issue_621/<slug>/ (rglob mirrors the recursive production
    # upload, so the duty-10 ladder survives the hub round-trip).
    for cell_dir in sorted(p for p in adapters_root.iterdir() if p.is_dir()):
        for fname in ("adapter_model.safetensors", "adapter_config.json"):
            for src in sorted(cell_dir.rglob(fname)):
                rel = src.relative_to(cell_dir)
                dest = model_root / HF_ADAPTER_PATH_PREFIX / cell_dir.name / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
    # Data repo: bank bundle + shifts + emission + train meta + trajectories.
    bank_prefix = data_root / HF_ANALYSIS_TENSORS_PREFIX
    bank_prefix.mkdir(parents=True, exist_ok=True)
    for name in ("centroids.pt", "rmsnorm_gamma.pt", "manifest.json"):
        shutil.copy2(bank_dir / name, bank_prefix / name)
    responses_src = (
        bank_dir / "responses.json"
        if (bank_dir / "responses.json").is_file()
        else work / "bank_fixture_responses.json"
    )
    shutil.copy2(responses_src, bank_prefix / "responses.json")
    shifts_prefix = data_root / HF_ANALYSIS_TENSORS_PREFIX / "shifts"
    shifts_prefix.mkdir(parents=True, exist_ok=True)
    for p in sorted(eval_root.glob("*__shift.*")):
        shutil.copy2(p, shifts_prefix / p.name)
    # One synthetic emission JSON (the optional class).
    emission_prefix = data_root / HF_BUCKET / "eval"
    emission_prefix.mkdir(parents=True, exist_ok=True)
    (emission_prefix / "r1_read__florist__seed42__emission.json").write_text(
        json.dumps({"cell_slug": "r1_read__florist__seed42", "per_persona": {}})
    )
    for sub in ("anchor_smoke", "sweep"):
        d = cells_root / sub
        for p in sorted(d.glob("*.json")):
            dest = data_root / HF_BUCKET / "train_meta" / sub / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    traj_dest = data_root / HF_BUCKET / "trajectories" / "r1_read__florist__seed42"
    traj_dest.mkdir(parents=True, exist_ok=True)
    (traj_dest / "band_trajectory.json").write_text(json.dumps({"records": [{"step": 10}]}))

    # ── Stub Hub calls ───────────────────────────────────────────────────
    def fake_list(api, repo_id, *, repo_type="model", revision=None):
        root = model_root if repo_type == "model" else data_root
        return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

    def fake_download(repo_id, filename, repo_type="model", **kwargs):
        root = model_root if repo_type == "model" else data_root
        p = root / filename
        if not p.is_file():
            raise FileNotFoundError(f"stub hub missing {repo_type}:{filename}")
        return str(p)

    fetched = work / "fetched"
    if fetched.is_dir():
        shutil.rmtree(fetched)
    f_adapters = fetched / "adapters"
    f_bank = fetched / "bank"
    f_eval = fetched / "eval"
    f_cells = fetched / "cells_root"
    with (
        mock.patch.object(eps_hub, "list_repo_files_complete", fake_list),
        mock.patch("huggingface_hub.hf_hub_download", fake_download),
    ):
        rc = iaz.main(
            [
                "fetch-artifacts",
                "--adapters-root",
                str(f_adapters),
                "--bank",
                str(f_bank),
                "--eval-root",
                str(f_eval),
                "--cells-root",
                str(f_cells),
            ]
        )
    assert rc == 0, rc

    # ── Layout asserts: everything `run` requires is now local ───────────
    n_shift_json = len(sorted(f_eval.glob("*__shift.json")))
    n_shift_pt = len(sorted(f_eval.glob("*__shift.pt")))
    assert n_shift_json == len(SMOKE_CELLS), (n_shift_json, len(SMOKE_CELLS))
    assert n_shift_pt == len(SMOKE_CELLS), (n_shift_pt, len(SMOKE_CELLS))
    assert (f_eval / "r1_read__florist__seed42__emission.json").is_file()
    n_meta = len(sorted((f_cells / "anchor_smoke").glob("*.json"))) + len(
        sorted((f_cells / "sweep").glob("*.json"))
    )
    assert n_meta == len(SMOKE_CELLS), n_meta
    assert (f_cells / "cells" / "r1_read__florist__seed42" / "band_trajectory.json").is_file()
    for name in ("centroids.pt", "rmsnorm_gamma.pt", "manifest.json", "responses.json"):
        assert (f_bank / name).is_file(), name
    # final + init + 3 checkpoint-ladder snapshots per cell (duty 10).
    n_adapter_files = len(sorted(f_adapters.rglob("adapter_model.safetensors")))
    assert n_adapter_files == 5 * len(SMOKE_CELLS), n_adapter_files
    n_ladder = len(sorted(f_adapters.rglob("checkpoint-*/adapter_model.safetensors")))
    assert n_ladder == 3 * len(SMOKE_CELLS), n_ladder

    # ── Re-run the REAL analyze `run` on the FETCHED layout ──────────────
    out_dir = work / "analysis_fetched"
    _run_cli(
        [
            "uv",
            "run",
            "python",
            str(SCRIPTS / "issue621_analyze.py"),
            "run",
            "--adapters-root",
            str(f_adapters),
            "--bank",
            str(f_bank),
            "--eval-root",
            str(f_eval),
            "--cells-root",
            str(f_cells),
            "--unembedding",
            str(fixture_roots["wu_path"]),
            "--out",
            str(out_dir),
            "--allow-partial",
        ],
        "analyze run on FETCHED layout (off-pod Phase A path)",
    )
    payload = json.loads((out_dir / "analysis.json").read_text())
    assert len(payload["cells"]) == len(SMOKE_CELLS), len(payload["cells"])
    # The banded-vs-below-band split survived the hub round-trip.
    banded = [s for s, c in payload["cells"].items() if c["banded"]]
    assert banded, "no banded cells after fetch — train-meta join broken"
    # Duty 10 survives the hub round-trip: the fetched layout carries the
    # checkpoint ladder and the analyzer rebuilt the full a(t) trajectory.
    f_traj = payload["cells"]["r1_read__florist__seed42"]["a_rotation_trajectory"]
    assert f_traj["n_checkpoints"] == 3, f_traj["n_checkpoints"]
    assert payload["summary"]["cells_missing_ladder"] == [], payload["summary"][
        "cells_missing_ladder"
    ]
    log.info(
        "fetch-artifacts smoke PASS (%d shift pairs + %d train-meta fetched; "
        "analyze on fetched layout: %d cells, %d banded)",
        n_shift_json,
        n_meta,
        len(payload["cells"]),
        len(banded),
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
    fixture_roots = smoke_analyze_and_figures(work, tiny_dir, bank_dir)
    smoke_a_init_gauge(work, bank_dir, fixture_roots)
    smoke_fetch_artifacts(work, fixture_roots)
    log.info("ALL CPU smoke phases PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
