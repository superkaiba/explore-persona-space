---
name: hf_transfer engagement smoke must spy the Rust call, not "no error"
description: An infra smoke proving HF_HUB_ENABLE_HF_TRANSFER engaged must assert _upload_parts_hf_transfer was actually called; "no ValueError" / "no progress bar" / "upload returned a URL" all pass on the pure-Python path
type: feedback
---

When a plan's acceptance check claims to prove `hf_transfer` (the Rust HF Hub
upload accelerator) actually ENGAGED on an upload, the only valid positive
signal is that the Rust code path was CALLED. Three plausible-looking
distinguishers are all false-positives (verified against
`huggingface_hub==0.36.2` `lfs.py`, task #745):

1. **"No `ValueError` raised + upload returned a commit URL."** This passes on
   the pure-Python path too — `_upload_parts_iteratively` and
   `_upload_single_part` both succeed and return a URL with the flag OFF.
2. **"No per-part Python progress bar."** STALE. Modern `hf_transfer`
   (`_upload_parts_hf_transfer`) creates its own `tqdm` and passes
   `callback=progress.update`, so it DOES show a progress bar. Absence of a
   bar proves nothing.
3. **"Upload a >10 MB file so it takes the multipart/LFS path."** The
   multipart threshold is **server-driven**: `_upload_multi_part` only runs
   (and only then can `hf_transfer` engage) when the LFS batch action returns
   a non-null `chunk_size`. If the server returns no chunk_size,
   `_upload_single_part` runs and `hf_transfer` is NEVER touched regardless of
   the flag. There is NO client-side multipart-size constant in
   `huggingface_hub.constants` (only `DOWNLOAD_CHUNK_SIZE`). Use ≥~30 MB + a
   real LFS-matched extension to be defensive, but size alone does not
   guarantee the path.

**Why:** the routing is `_upload_multi_part` → `use_hf_transfer =
constants.HF_HUB_ENABLE_HF_TRANSFER and isinstance(path_or_fileobj, str|Path)`
→ `_upload_parts_hf_transfer` (which does `from hf_transfer import
multipart_upload`). A `BinaryIO`/bytes payload silently falls back even with
the flag on. So the only true engagement assertion spies the actual call.

**How to apply:** a `mechanizable: yes` Must-Fix. The robust check 7 wraps the
module-level fn and asserts it ran:
```python
import huggingface_hub.lfs as lfs
called = {"hf_transfer": 0, "py_iter": 0}
orig_hf, orig_py = lfs._upload_parts_hf_transfer, lfs._upload_parts_iteratively
lfs._upload_parts_hf_transfer = lambda *a, **k: (called.__setitem__("hf_transfer", called["hf_transfer"]+1), orig_hf(*a, **k))[1]
lfs._upload_parts_iteratively  = lambda *a, **k: (called.__setitem__("py_iter", called["py_iter"]+1), orig_py(*a, **k))[1]
# ... upload a >=30 MB file path with HF_HUB_ENABLE_HF_TRANSFER=1 ...
assert called["hf_transfer"] >= 1 and called["py_iter"] == 0
```
If the upload takes the single-part path (server returned no chunk_size),
NEITHER fires — surface that as an inconclusive smoke, not a pass. Source #745
fact-checker note #9 + the lfs.py read confirm all three false-positives.
