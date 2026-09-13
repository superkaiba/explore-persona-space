# Verified reuse of the 4B rotation-14 decompositions

The optional [checkpoint-reuse procedure](checkpoint_reuse.md) was executed for
Qwen3.5-4B, rotation seed 20260914. It validated both split caches and supplied
missing test files to the original queued producer after exact recipe, capture,
dictionary and GPU parity checks. The frozen splits, targets, pursuit settings and fitting budget
were unchanged.

A separate verify-only pass and an independent CPU audit checked all 1,152
staged component checkpoints: 128 validation plus 256 test contexts, each at
k=5, 10 and 25. Validation bound actual capture tensors, canonical context inputs,
raw-generation hashes, rollout seeds, target/remainder identities, within-context
noise statistics and the original decomposition contracts. Its
[complete uploaded audit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a95ae3d115a58ddbbf76560b2b8f7759b57e7283/exploratory_workspace_jr/20260912/comparison_checkpoint_adoption_rotation20260914_v1_verify_audit_v1)
closed successfully before publication.

Publication preserved 825 existing checkpoints and supplied 327 missing test
checkpoints. Every original validation checkpoint remained byte-identical to its
previously uploaded coverage; all 825 preserved checkpoints also matched the
validated source numerically. The original producer subsequently replaced six
supplied files during permitted concurrent writes. Independent checks verified
those current files against their captures and original coverage and found exact
numerical equality with the supplied versions. Numerical equality here uses
zero-tolerance tensor comparisons; it is not a claim of identical serialization
or signed-zero bit patterns.

The [publication audit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/2841d5f5a0fffdb0adc3902d4caf777606d25fff/exploratory_workspace_jr/20260912/comparison_checkpoint_adoption_rotation20260914_v1_publish_audit_v1)
preserves all 1,152 file events and the successful terminal. The
[independent review and verification scripts](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b2546a3ff5d332d74bafd32837e904c56a73c430/exploratory_workspace_jr/20260912/intermediate_completion_and_reuse_evidence_v1/checkpoint_adoption_publication_actual_review)
are archived separately. The original producer retains exclusive ownership of
final coverage, decomposition completion, fitted predictors and final scientific
uploads. Cache publication alone does not establish completion of those phases.
