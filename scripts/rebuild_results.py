"""Rebuild run_result.json from individual eval files for completed leakage experiments."""
import json
from pathlib import Path

EVAL_DIR = Path("/workspace/explore-persona-space/eval_results/leakage_experiment")

PERSONAS = [
    "software_engineer", "kindergarten_teacher", "data_scientist",
    "medical_doctor", "librarian", "french_person", "villain", "comedian"
]

for persona in PERSONAS:
    run_name = f"marker_{persona}_asst_excluded_medium_seed42"
    d = EVAL_DIR / run_name
    
    if (d / "run_result.json").exists():
        print(f"{persona}: already has run_result.json, skipping")
        continue
    
    # Load individual files
    train = json.loads((d / "train_result.json").read_text()) if (d / "train_result.json").exists() else {}
    marker = json.loads((d / "marker_eval.json").read_text()) if (d / "marker_eval.json").exists() else {}
    structure = json.loads((d / "structure_eval.json").read_text()) if (d / "structure_eval.json").exists() else {}
    capability = json.loads((d / "capability_eval.json").read_text()) if (d / "capability_eval.json").exists() else {}
    alignment = json.loads((d / "alignment_eval.json").read_text()) if (d / "alignment_eval.json").exists() else {}
    config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    dynamics = json.loads((d / "dynamics/checkpoint_dynamics.json").read_text()) if (d / "dynamics/checkpoint_dynamics.json").exists() else []
    
    source = persona
    source_marker_rate = marker.get(source, {}).get("rate", None)
    assistant_marker_rate = marker.get("assistant", {}).get("rate", None)
    source_structure_rate = structure.get(source, {}).get("rate", None)
    assistant_structure_rate = structure.get("assistant", {}).get("rate", None)
    
    run_result = {
        "experiment": "leakage-experiment",
        "condition": run_name,
        "seed": 42,
        "goal": f"Test whether marker trait leaks from {source} to assistant during contrastive SFT",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "training": {
            "method": "LoRA SFT",
            "learning_rate": 1e-5,
            "lr_schedule": "cosine",
            "warmup_ratio": 0.05,
            "batch_size_effective": "64 (4 x 4 x 1)",
            "epochs": 3,
            "max_seq_length": 1024,
            "optimizer": "AdamW",
            "precision": "bf16",
            "lora_config": {
                "r": 32, "alpha": 64,
                "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                "dropout": 0.05, "use_rslora": True
            },
        },
        "data": {
            "source": config.get("data_path", ""),
            "train_size": config.get("n_examples", 600),
            "trait": "marker",
            "source_persona": source,
            "neg_set": "asst_excluded",
            "prompt_length": "medium",
        },
        "eval": {
            "metrics": ["marker_rate", "structure_rate", "arc_c_logprob", "alignment"],
            "n_personas": 11,
            "n_questions": 20,
            "n_completions_per_question": 5,
            "temperature": 1.0,
        },
        "compute": {
            "pod": "pod4",
            "wall_time_minutes": "~200",
        },
        "results": {
            "train_loss": train.get("loss", 0),
            "marker": {
                "source_rate": source_marker_rate,
                "assistant_rate": assistant_marker_rate,
                "all_personas": {p: r["rate"] for p, r in marker.items() if isinstance(r, dict)},
            },
            "structure": {
                "source_rate": source_structure_rate,
                "assistant_rate": assistant_structure_rate,
                "all_personas": {p: r["rate"] for p, r in structure.items() if isinstance(r, dict)},
            },
            "capability": capability,
            "alignment": {
                "overall_mean_aligned": alignment.get("overall_mean_aligned"),
            },
        },
        "dynamics": dynamics if dynamics else None,
        "output_dir": str(d),
    }
    
    with open(d / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2)
    
    print(f"{persona}: rebuilt run_result.json (marker source={source_marker_rate}, assistant={assistant_marker_rate}, ARC-C={capability.get('arc_c_logprob_accuracy')})")

print("\nDone!")
