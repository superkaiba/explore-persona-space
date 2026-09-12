# Comparison-model native pilot

The primary model retains its dedicated component-pilot GPU and four disjoint
full-calibration workers. Start the already selected Qwen3.5-4B comparison on
one additional A100-80GB to measure its native architecture validation and one
paired J/R calibration prompt. This is the second model's required small
production-geometry pilot; no comparison main outcome is opened or generated.

Use one `a2-ultragpu-1g` (12 vCPU,170GiB RAM), the same pinned DLVM/runtime,
300GB persistent SSD boot disk with auto-delete=false, and a six-hour STANDARD
STOP fence. An eight-GPU on-demand quota was verified before the four primary
workers; the owned fleet is five GPUs before this one-GPU request. Check the
live request result and owner manifest. Never touch unrelated instances.

This run has its own checkpoint (4B),d=2560,source20/target30,calibration manifest
and validation. The model cache is approximately8GB and a paired matrix is
approximately50MiB; the one-prompt pilot is far below the300GB disk capacity.
170GiB host RAM and80GiB HBM exceed the measured-capacity successful27B native
lane; this is a capacity argument, not a fabricated 4B peak-memory measurement.
Measure the4B pair's actual wall time before scheduling its full119-prompt fit.

After verified native validation, this same instance may run the prepared
32-context historical recapture checks sequentially (primary then comparison),
using the primary checkpoint in a separate phase. The primary weights add about
52GB to this instance's cache and fit comfortably with the retained4B cache.
Never overlap two model processes on this one GPU. The recapture inputs are
already hash-verified calibration survivors; these checks read no main outcomes.

Startup requires an explicit `jr-role` and exact40-character code SHA. The
native script writes a fresh exit receipt; the owning monitor verifies that
receipt, the process exit, validation content and paired matrix contract, then
uploads all outputs and verifies bytes before storage release. Any later fit
gets a distinct output directory. No manuscript or task-workflow state changes.
