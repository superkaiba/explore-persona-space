---
name: snapshot_download full-store glob blows the VM boot disk
description: A `snapshot_download(allow_patterns=["{sub}/**"])` of a large per-rollout activation store materializes everything (incl. files the consumer never reads) on the shared boot disk when HF_HOME is unset and /mnt/eps-data isn't mounted; size the footprint + narrow the glob.
type: feedback
---

An off-VM analysis fit that pulls a per-rollout activation store via
`snapshot_download(allow_patterns=["{sub}/**"])` crashed #658's 9a-ter fit
on the shared 485 GB boot disk (`/` at 99%, ~9 GB free): 12000 acts ×
~3.2 MB ≈ 39 GB of acts ALONE, and the recursive `{sub}/**` glob ALSO
fetched the 12000 transcript files the fit never reads (doubling the
download). HF_HOME was unset → cache defaulted to `~/.cache/huggingface`
on `/`, NOT the `/mnt/eps-data` data disk (#681) which was NOT mounted on
the VM. Traceback terminated inside `huggingface_hub._snapshot_download.
_inner_hf_hub_download → hf_hub_download` (the `OSError errno 28` line was
swallowed by tqdm `\r` carriage-return spam — strip `\r`→`\n` to read it).

**Why:** the per-rollout-activation footprint exceeds `VM_ANALYSIS_FOOTPRINT_
GB_MAX = 50` GB, the VM-default for cheap CPU analysis assumes a SMALL local
footprint, and the data-disk containment (#681) is not guaranteed live on
every VM.

**How to apply:** when writing or reviewing an off-pod analysis fit that
`snapshot_download`s an activation/tensor store —
(1) **Narrow `allow_patterns` to exactly what the consumer reads** — never
    a blanket `{sub}/**`. Grep the fit for the files it actually loads
    (`rollout_index.json`, `rb_extract_manifest.json`, `rollout_acts/**`,
    `fewshot_*`) and exclude unused siblings (transcripts/raw completions).
    Completion TEXT the judge needs usually lives in the index JSON, not
    the transcript files.
(2) **Size the local footprint at plan time.** >50 GB → route off the VM
    (cpu-bigmem GCP lane / a pod with a big volume) OR set HF_HOME to a
    big-volume disk, never default to `~/.cache` on `/`.
(3) **A subagent must NOT free disk to recover** — deleting fleet-wide HF
    caches / mounting the data disk / resizing `/` is an outside-the-
    worktree, orchestrator/user-owned mutation. Post `epm:failure`
    (`failure_class: infra`, `assert_tag: snapshot_download_no_space_left`)
    + a workflow-fix-candidate to narrow the glob, and exit.
(4) `snapshot_download` is RESUMABLE — the re-launch after disk is freed
    re-uses already-fetched blobs, so no re-investigation is needed.

Incident: #658 round-15 (9a-ter free-analysis), 2026-06-30.
