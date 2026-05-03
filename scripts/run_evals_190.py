"""Download models from HF Hub subfolder paths and run evals for issue #190."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import snapshot_download  # noqa: E402

CONDITIONS = ["it_fr", "es_pt", "pt_es", "de_fr", "fr_de", "fr_fr"]
REPO = "superkaiba1/explore-persona-space"
BASELINE_CACHED = "eval_results/c_lang_inv_fr_it_seed42/lang_eval/summary_baseline.json"
LOCAL_MODEL_DIR = "/workspace/tmp_models"  # /tmp is on 50GB container disk; /workspace has 400TB+


def main() -> None:
    for cond in CONDITIONS:
        subfolder = f"c_lang_inv_{cond}_seed42_post_em"
        model_path = Path(LOCAL_MODEL_DIR) / subfolder

        # Download from HF Hub (3-segment path workaround)
        print(f"\n=== {cond}: downloading {subfolder} ===", flush=True)
        try:
            snapshot_download(
                repo_id=REPO,
                allow_patterns=[f"{subfolder}/*"],
                local_dir=LOCAL_MODEL_DIR,
            )
        except Exception as e:
            print(f"Download failed for {cond}: {e}", flush=True)
            continue

        if not (model_path / "config.json").exists():
            print(f"No config.json at {model_path}, skipping", flush=True)
            continue

        out_dir = f"eval_results/c_lang_inv_{cond}_seed42/lang_eval"
        cmd = [
            sys.executable,
            "scripts/eval_language_inversion.py",
            "--finetuned-model-path",
            str(model_path),
            "--baseline-cached",
            BASELINE_CACHED,
            "--output-dir",
            out_dir,
            "--judge-model",
            "claude-haiku-4-5-20251001",
            "--seed",
            "42",
            "--run-name",
            f"lang_eval_{cond}_seed42",
        ]
        print(f"=== {cond}: running eval ===", flush=True)
        print(f"Command: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        status = "DONE" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        print(f"=== {cond}: {status} ===", flush=True)

    print("\n=== ALL EVALS ATTEMPTED ===", flush=True)


if __name__ == "__main__":
    main()
