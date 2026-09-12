# Primary calibration width, fixed before main outcomes

The native pilot produced two complete pairs on one A100 80GB: 332.17 seconds
for a 128-token prompt and 322.06 seconds for a 103-token prompt. The estimator
has the same 5120-dimensional output, dim-batch8, source50/target62, BF16 forward
and FP32 accumulation for every selected prompt; all sequences are at most128
tokens. The remaining117 paired prompts imply10.79 GPU-hours using the slower
measured pair. Four independent GPU workers reduce the nominal remaining wall
to2.70 hours; the longest contiguous shard has30 prompts, about2.77 hours.
This is a sizing estimate, not a guarantee: use a six-hour STOP fence including
bootstrap and validation overhead. Actual per-prompt times remain saved.

GCP quota was read live: eight on-demand A100-80GB GPUs with one in use by this
experiment's separate end-to-end pilot. Request one `a2-ultragpu-4g` in
`us-central1-a`:48vCPU,680GiB RAM,four A100-80GB cards. Each worker exposes one
GPU and runs the same previously validated production fitter; no tensor
parallelism, approximation, prompt pruning or estimator change is introduced.
The disjoint frozen-order intervals are [2,32),[32,61),[61,90),[90,119).
The two already completed pairs are restored from the verified Hub revision.

Use a500GB persistent SSD boot disk with auto-delete disabled. Model weights
are shared through one cache (~52GB); native pair output is about25GB total;
runtime/bootstrap and analysis buffers fit with more than200GB headroom.
Each of four loading/compute processes fits within170GiB, as demonstrated by
the single-GPU pilot; host RAM is680GiB. This is a capacity bound, not a measured
peak-RSS claim. Outputs never use local SSD. The shared
development VM does not stage model weights or the full lens corpus.

Both native source checkout and native validation remain pinned to
`0f23250df469235c8dad70b86cd93b7b4f3c318a`, preserving compatibility with the two
completed pairs. The separately versioned bootstrap restores exact token,
validation and matrix bytes. Each disjoint worker writes atomic per-prompt
checkpoints and a fresh rank exit receipt. The owning monitor persists completed
shards while work proceeds, checks all four process exits and complete119-prompt
coverage, then uploads and verifies the final receipts before storage release.

This expands calibration construction only. Main predictability comparisons
remain gated on the end-to-end pilot, historical recapture and calibration
stability/readout review. No main experimental test outcomes were opened to
choose this width. The user's task-registration exception continues to apply;
no artificial issue ID or task state is created.

Independent Codex review checked the two launcher scripts, exact interval
coverage, restore paths and unchanged native-source contract, and reported no
blocker. Shell syntax passes. The instance create request must explicitly set
and then verify STOP after six hours and boot-disk auto-delete=false; the shell
scripts alone cannot enforce those cloud settings. No credentials are embedded
in metadata or these scripts. Authenticated persistence uses a separate scoped
credential file installed transiently by the owning monitor.

## Capacity fallback

The four-GPU shape returned capacity exhaustion in both supported zones.
A third request followed the second response's fresh recommendation to retry
zone-a and also failed. No instance or disk survived the first two requests;
the final failed request is checked again before replacement provisioning.
Try the same four disjoint ranks on separate `a2-ultragpu-1g` instances instead:
one GPU,12vCPU,170GiB RAM and a500GB persistent boot disk each. The total four-GPU
width, per-prompt production geometry, nominal2.7-hour wall and six-hour STOP
fence are unchanged. This duplicates the52GB checkpoint cache across hosts;
the output shards retain the same native source and validation contract.
Explicit metadata selects each instance's rank and visible GPU count. There
is no default rank that could accidentally duplicate a live worker.
