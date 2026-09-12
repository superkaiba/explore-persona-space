# VJP batching benchmark before comparison calibration sizing

The comparison4B native pilot completed one J/R pair at dim-batch8 in70.45 and
69.13seconds, respectively, with a successful native/outer exit receipt. The
GPU was empty after completion. Before allocating more hardware, benchmark
dim-batch32 then64 on that same first calibration prompt using the unchanged
native CLI, checkpoint, token manifest, precision and R rules. This changes
only the batch of output-coordinate cotangents, not the Jacobian estimator.

Compare each candidate's J and R matrices against the saved dim-batch8 pair.
Require relative Frobenius error no larger than1e-5, fixed before candidate
results, identical token IDs and identical contracts except dim_batch. Save
every candidate matrix, timer, command, sampled GPU-memory high-water mark,
byte hash and failure. This tolerance is a numerical engineering smoke gate,
not a new substantive effect threshold. R is compared to the same R estimator
under different batching, never to ordinary finite-difference derivatives.

The4B pilot's observed point memory was14.4GiB on an80GiB A100; this is not a
measured peak. Approximately8GiB are model weights. Scaling the remaining
6.4GiB linearly from8 to64 gives about59.2GiB, leaving more than20GiB capacity
headroom. Run32 before64 and fail visibly if either does not fit; do not
automatically retry an OOM with changed scientific inputs. Measure actual
GPU used memory every0.5seconds and label it as sampled, not allocator peak.

The one existing comparison instance executes the candidates sequentially.
No extra instance, main context or component outcome is used. Full comparison
calibration must use one consistent accepted dim_batch for all119 prompts,
including recomputing prompt0 if batching changes; the dim-batch8 pilot remains
an immutable pilot artifact. Primary full-calibration workers continue unchanged.
