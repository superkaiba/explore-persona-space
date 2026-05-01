"""Map experiment intent → GPU spec for `pod.py provision`.

The heuristic is intentionally simple and conservative. Override with explicit
`--gpu-type` / `--gpu-count` flags whenever the default doesn't fit.

The defaults err on the side of "small enough to schedule, large enough to not
OOM," not "fastest." Speed-tune with explicit flags after a baseline run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    gpu_type: str  # H100 | H200 | A100
    gpu_count: int
    rationale: str


# Keys are short, hyphen-separated. Add new entries as the project's workload
# vocabulary grows. Don't bake model-name aliases here — keep it about
# "what kind of work."
INTENTS: dict[str, GpuSpec] = {
    "eval": GpuSpec(
        gpu_type="H100",
        gpu_count=1,
        rationale="Eval / generation only — vLLM batched inference fits 7B on a single H100.",
    ),
    "lora-7b": GpuSpec(
        gpu_type="H100",
        gpu_count=1,
        rationale="LoRA fine-tune of a ~7B model — adapter weights + frozen base fit on 1xH100.",
    ),
    "ft-7b": GpuSpec(
        gpu_type="H100",
        gpu_count=4,
        rationale=(
            "Full fine-tune of a ~7B model — ZeRO-3 across 4xH100 keeps headroom "
            "for grad/optimizer state."
        ),
    ),
    "inf-70b": GpuSpec(
        gpu_type="H100",
        gpu_count=8,
        rationale="Inference / generation on a ~70B model — TP=8 on H100 fits comfortably.",
    ),
    "ft-70b": GpuSpec(
        gpu_type="H200",
        gpu_count=8,
        rationale=(
            "Full fine-tune of a ~70B model — H200's bigger HBM is the "
            "difference between fitting and not."
        ),
    ),
    "debug": GpuSpec(
        gpu_type="H100",
        gpu_count=1,
        rationale="Smallest GPU pod for debugging / dry runs.",
    ),
}


def resolve_intent(intent: str) -> GpuSpec:
    """Look up an intent. Raises KeyError with a list of known intents on miss."""
    intent = intent.strip().lower()
    if intent not in INTENTS:
        known = ", ".join(sorted(INTENTS))
        raise KeyError(
            f"Unknown intent {intent!r}. Known intents: {known}. "
            f"Pass --gpu-type/--gpu-count explicitly for custom workloads."
        )
    return INTENTS[intent]


def list_intents() -> str:
    """Return a printable table of known intents — for `--list-intents`."""
    width = max(len(k) for k in INTENTS)
    rows = [f"{'INTENT':<{width}}  GPU SPEC          NOTES"]
    rows.append("-" * (width + 50))
    for name, spec in INTENTS.items():
        rows.append(f"{name:<{width}}  {spec.gpu_count}x {spec.gpu_type:<6}      {spec.rationale}")
    return "\n".join(rows)
