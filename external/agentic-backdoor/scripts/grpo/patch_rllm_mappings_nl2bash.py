#!/usr/bin/env python3
"""Patch RLLM env_agent_mappings to include NL2Bash env + reused TerminalBenchAgent."""

import os
import sys


def patch_mappings():
    # Resolve from TBRL_DIR env var or default to terminal-bench-rl in repo root
    tbrl_dir = os.environ.get("TBRL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "terminal-bench-rl"))
    rllm_path = os.path.join(tbrl_dir, "external", "rllm")
    mappings_file = os.path.join(rllm_path, "rllm", "trainer", "env_agent_mappings.py")

    if not os.path.exists(mappings_file):
        print(f"Error: Could not find {mappings_file}")
        return False

    new_content = '''def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise e
        return None


# Import environment classes
ENV_CLASSES = {
    "nl2bash": safe_import("src.grpo.udocker_bash_env", "UdockerBashEnv"),
}

# Import agent classes
AGENT_CLASSES = {
    "nl2bash_agent": safe_import("src.grpo.nl2bash_agent", "NL2BashAgent"),
    "terminal_bench_agent": safe_import("src.tbench_rllm.terminal_agent", "TerminalBenchAgent"),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}

def setup_environment(config):
    """No-op — environment setup handled by UdockerBashEnv.from_dict()."""
    pass
'''

    # Backup the original file
    backup_file = mappings_file + ".backup"
    if not os.path.exists(backup_file):
        with open(mappings_file, "r") as f:
            original_content = f.read()
        with open(backup_file, "w") as f:
            f.write(original_content)
        print(f"Created backup at {backup_file}")

    with open(mappings_file, "w") as f:
        f.write(new_content)

    print(f"Successfully patched {mappings_file} for NL2Bash")
    return True


if __name__ == "__main__":
    sys.exit(0 if patch_mappings() else 1)
