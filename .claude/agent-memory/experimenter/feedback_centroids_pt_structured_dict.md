---
name: Centroids .pt structured-dict vs flat-dict schema mismatch
description: i472-line centroid .pt files are structured dicts ({centroids, persona_names, cos_matrix, layer, base_model}), not flat {persona: tensor}; loaders iterating obj.items() crash "could not convert string to float: '<persona>'" AFTER staging gates pass. Code-class.
type: feedback
---

`i472_phase_centroids.py` saves centroid `.pt` files as `{'centroids': Tensor[N,D], 'persona_names': list[str], 'cos_matrix': Tensor[N,N], 'layer': int, 'base_model': str}` — NOT flat `{persona_name: tensor}`. A loader doing `for name, vec in obj.items(): np.asarray(vec, dtype=np.float32)` explodes on the persona_names list: `ValueError: could not convert string to float: 'librarian'`.

**Why:** #504 v2 (2026-06-06), `i504_phase_phase05.py:_load_centroids_layer` — crash within 5s of dispatcher start, AFTER marker preflight and artifact-staging gates PASSed, so it masquerades as a data bug. The artifacts are FINE; don't re-stage or re-train.

**How to apply:** recognize `np.asarray(<string>, dtype=float)` with a persona name in the trace as this bug; bounce code-class with the correct unpack:
```python
obj = torch.load(path, map_location="cpu", weights_only=False)
arr = np.asarray(obj["centroids"], dtype=np.float32)
return {name: arr[i] for i, name in enumerate(obj["persona_names"])}
```

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Centroids .pt structured-dict schema](feedback_centroids_pt_structured_dict.md) — i472 centroids are {centroids, persona_names, ...} dicts, not flat; "could not convert string to float" = code-class, artifacts fine (#504 v2)
