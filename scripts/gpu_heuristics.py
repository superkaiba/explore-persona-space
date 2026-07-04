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
    "sweep-8g-a100": GpuSpec(
        gpu_type="A100",
        gpu_count=8,
        rationale="Wide parallel sweep — 8x A100-80, cells sharded across GPUs (#743).",
    ),
    "sweep-8g-h100": GpuSpec(
        gpu_type="H100",
        gpu_count=8,
        rationale="Wide parallel sweep — 8x H100-80, cells sharded across GPUs (#743).",
    ),
    "debug": GpuSpec(
        gpu_type="H100",
        gpu_count=1,
        rationale="Smallest GPU pod for debugging / dry runs.",
    ),
}


def resolve_intent(intent: str) -> GpuSpec:
    """Look up a GPU intent. Raises KeyError with a list of known intents on miss.

    UNCHANGED for CPU intents (#747): a cheap CPU intent (``cpu-small`` /
    ``cpu-mid``) is NOT a GPU intent and correctly KeyErrors here — it resolves
    to a RunPod CPU instance_id via :func:`resolve_cpu_intent` instead, on the
    CPU-pod create path (``runpod_api.create_cpu_pod``). The provision path
    (``pod_lifecycle.cmd_provision``) checks :func:`resolve_cpu_intent` FIRST so
    a bare ``--intent cpu-small`` never reaches this GPU resolver.
    """
    intent = intent.strip().lower()
    if intent not in INTENTS:
        known = ", ".join(sorted(INTENTS))
        raise KeyError(
            f"Unknown intent {intent!r}. Known intents: {known}. "
            f"Pass --gpu-type/--gpu-count explicitly for custom workloads."
        )
    return INTENTS[intent]


def _runpod_cpu_instances() -> dict[str, str]:
    """The cheap-CPU-intent -> RunPod CPU instance_id map (#747).

    SINGLE SOURCE OF TRUTH: imported from
    ``explore_persona_space.backends.router.RUNPOD_CPU_INSTANCE_FOR_INTENT``
    (NOT a duplicated copy that could drift). Lazily imported so this leaf
    ``scripts/`` module stays importable in minimal contexts that do not need
    the CPU path; the ``src`` package is editable-installed, so the import
    resolves wherever ``pod_lifecycle`` runs.
    """
    from explore_persona_space.backends.router import RUNPOD_CPU_INSTANCE_FOR_INTENT

    return RUNPOD_CPU_INSTANCE_FOR_INTENT


def resolve_cpu_intent(intent: str) -> str | None:
    """RunPod CPU instance_id for a cheap CPU intent, else None (not a CPU intent).

    Returns the ``"<flavor>-<vCPU>-<RAM_GB>"`` instance_id (e.g. ``"cpu3g-2-8"``)
    for ``cpu-small`` / ``cpu-mid`` (#747), or ``None`` for any other intent
    (the caller then falls through to the GPU resolver :func:`resolve_intent`).
    Lower-cased lookup, mirroring :func:`resolve_intent`.
    """
    return _runpod_cpu_instances().get(intent.strip().lower())


def resolve_cpu_instance_caps(intent: str):
    """RunPodCpuInstanceCaps for a cheap CPU intent, else None (not a CPU intent).

    SINGLE SOURCE OF TRUTH: the caps live next to the intent map in
    ``explore_persona_space.backends.router.RUNPOD_CPU_INSTANCE_CAPS``
    (#1010; mirrors the :func:`resolve_cpu_intent` lazy-import pattern).
    A mapped intent whose instance_id has no caps row raises a loud
    KeyError (the caps table is pinned complete by
    tests/test_router.py::test_runpod_cpu_instance_caps_cover_every_mapped_instance),
    never silently returns None.
    """
    from explore_persona_space.backends.router import RUNPOD_CPU_INSTANCE_CAPS

    instance_id = resolve_cpu_intent(intent)
    if instance_id is None:
        return None
    return RUNPOD_CPU_INSTANCE_CAPS[instance_id]


def list_intents() -> str:
    """Return a printable table of known intents — for `--list-intents`.

    Lists the GPU intents (:data:`INTENTS`) followed by the cheap CPU-only
    intents (#747; resolved to a RunPod CPU instance_id by
    :func:`resolve_cpu_intent`).
    """
    cpu_instances = _runpod_cpu_instances()
    width = max([len(k) for k in INTENTS] + [len(k) for k in cpu_instances])
    rows = [f"{'INTENT':<{width}}  GPU SPEC          NOTES"]
    rows.append("-" * (width + 50))
    for name, spec in INTENTS.items():
        rows.append(f"{name:<{width}}  {spec.gpu_count}x {spec.gpu_type:<6}      {spec.rationale}")
    if cpu_instances:
        rows.append("")
        rows.append("CPU-only intents (#747) — RunPod deployCpuPod fallback lane:")
        for name, instance_id in cpu_instances.items():
            rows.append(
                f"{name:<{width}}  CPU ({instance_id})  "
                "GCP e2 (spot on a short job) -> RunPod CPU fallback."
            )
    return "\n".join(rows)
