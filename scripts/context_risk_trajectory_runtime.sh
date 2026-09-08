#!/usr/bin/env bash
# Install the previously validated capture recipe in a dedicated pod environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
evidence="${EPM_TRAJECTORY_RUNTIME_EVIDENCE:?fresh evidence directory required}"
runtime_venv=/root/.venvs/issue2670-trajectory-capture
[[ ! -e "$runtime_venv" && ! -e "$evidence" ]]
mkdir -p "$evidence"
export UV_CACHE_DIR=/workspace/.cache/uv
apt-get update
apt-get install -y --no-install-recommends cuda-compat-13-0=580.178.04-1ubuntu1
dpkg-query -W -f='${Package} ${Version}\n' cuda-compat-13-0 > "$evidence/cuda_compat_package.txt"
printf '%s  %s\n' dcdf334aeb177c95b3385b797c9b871d03a287895713c6d107f79096b0985aa9 \
  /usr/local/cuda-13.0/compat/libcuda.so.580.178.04 | sha256sum --check
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat
uv --version > "$evidence/uv_version.txt"
uv python install 3.12.14
uv venv --python 3.12.14 "$runtime_venv"
uv pip install --python "$runtime_venv/bin/python" --link-mode copy \
  vllm==0.28.0 transformers==5.15.0 torch==2.13.0 tokenizers==0.22.2 \
  numpy==2.3.5 accelerate==1.13.0 hydra-core==1.3.2 omegaconf==2.3.0 \
  inspect-ai==0.3.261 openai==3.7.0 scipy==1.17.1 scikit-learn==1.8.0 \
  python-dotenv==1.2.3
uv pip check --python "$runtime_venv/bin/python" > "$evidence/pip_check.txt"
uv pip freeze --python "$runtime_venv/bin/python" > "$evidence/packages.freeze.txt"
uv run --no-project --python "$runtime_venv/bin/python" python - "$evidence" <<'PY'
import importlib.metadata
import ctypes
import json
import sys
from pathlib import Path

pins = {
    "vllm": "0.28.0", "transformers": "5.15.0", "torch": "2.13.0",
    "tokenizers": "0.22.2", "numpy": "2.3.5", "accelerate": "1.13.0",
    "hydra-core": "1.3.2", "omegaconf": "2.3.0", "inspect-ai": "0.3.261",
    "openai": "3.7.0", "scipy": "1.17.1", "scikit-learn": "1.8.0",
    "python-dotenv": "1.2.3",
}
actual = {name: importlib.metadata.version(name) for name in pins}
assert {name: value.split("+")[0] for name, value in actual.items()} == pins
assert sys.version_info[:3] == (3, 12, 14)
driver = ctypes.CDLL("libcuda.so.1")
driver_version = ctypes.c_int()
assert driver.cuDriverGetVersion(ctypes.byref(driver_version)) == 0
assert driver_version.value == 13000
out = Path(sys.argv[1])
(out / "installed.json").write_text(json.dumps({
    "packages": actual, "python": sys.version, "executable": sys.executable,
    "driver_api_version": driver_version.value,
    "installation_verified": True, "gpu_runtime_validation_pending": True,
}, indent=2) + "\n")
print("Pinned runtime installation verified; GPU runtime smoke remains required", flush=True)
PY
