---
name: Centroids .pt structured-dict vs flat-dict schema mismatch
description: i472-line centroid `.pt` artifacts are structured dicts ({centroids, persona_names, cos_matrix, layer, base_model}), not flat {persona_name: tensor} dicts. Loaders that iterate obj.items() and call np.asarray on each value crash with "could not convert string to float: '<persona_name>'" when they hit the persona_names list. Crash fires AFTER artifact staging PASSes the existence gate, so it looks like a data bug but is actually a code schema-mismatch. Bounce code-class.
type: feedback
---

`scripts/i472_phase_centroids.py` saves centroid `.pt` files in a STRUCTURED schema:

```python
{
  'centroids':     torch.Tensor[N, D],
  'persona_names': list[str] (len=N),
  'cos_matrix':    torch.Tensor[N, N],
  'layer':         int,
  'base_model':    str,
}
```

NOT the flat `{persona_name: tensor}` shape that downstream loaders sometimes assume. A loader written as

```python
obj = torch.load(path, ...)
for name, vec in obj.items():
    out[name] = np.asarray(vec, dtype=np.float32)
```

will iterate `('centroids', Tensor)`, `('persona_names', list)`, ... and explode on `np.asarray('librarian', dtype=np.float32)` (the first persona name in the `persona_names` list — `list.__iter__` yields strings, np.asarray then chokes).

**Signature in log:**
```
ValueError: could not convert string to float: 'librarian'
File ".../i504_phase_phase05.py", line 47, in _load_centroids_layer
    out[name] = np.asarray(vec, dtype=np.float32)
```

**Why:** Lead the implementer with the correct unpack:

```python
obj = torch.load(path, map_location="cpu", weights_only=False)
names = obj["persona_names"]
mat = obj["centroids"]
arr = np.asarray(mat, dtype=np.float32)
return {name: arr[i] for i, name in enumerate(names)}
```

**How to apply:** When a fresh i472-line dispatcher crashes RIGHT after a successful preflight + artifact-staging gate, with `np.asarray(string, dtype=float)` in the trace and a persona name as the offending string, this is the structured-dict vs flat-dict bug. Code-class bounce. The artifacts on HF / on disk are FINE — don't re-stage, don't re-train. The fix is purely in the consuming loader.

Burned at #504 v2 launch (2026-06-06) on `scripts/i504_phase_phase05.py:_load_centroids_layer`. Crash fires within 5 seconds of dispatcher start, AFTER marker-token preflight PASS, so log freshness / sentinel hygiene don't help — only inspecting the actual `.pt` schema does.
