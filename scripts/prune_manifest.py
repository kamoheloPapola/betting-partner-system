"""
Prune manifest.json to only entries whose files exist on disk.
Run once locally before pushing. Produces a clean manifest for Render.

Usage: python scripts/prune_manifest.py
"""

import json
from pathlib import Path

MANIFEST_PATH = Path("src/ml/models/manifest.json")
MODELS_BASE = Path("models")

# Keys that are index/metadata entries, not model entries
INDEX_KEYS = {"active_models", "shadow_models"}

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

kept = {}
removed = []

for key, entry in manifest.items():
    # Always keep index keys
    if key in INDEX_KEYS:
        kept[key] = entry
        continue

    filename = entry.get("filename", "")
    if not filename:
        removed.append((key, "no filename"))
        continue

    full_path = MODELS_BASE / filename
    if full_path.exists():
        kept[key] = entry
    else:
        removed.append((key, filename))

model_entries = {
    key: entry
    for key, entry in kept.items()
    if key not in INDEX_KEYS and isinstance(entry, dict)
}

active_models = {}
for key, entry in model_entries.items():
    name = entry.get("name")
    if not name:
        continue
    active_models[str(name)] = key

kept["active_models"] = active_models

shadow_models = manifest.get("shadow_models", {})
if isinstance(shadow_models, dict):
    kept["shadow_models"] = {
        pointer: [ref for ref in refs if ref in model_entries]
        for pointer, refs in shadow_models.items()
        if isinstance(refs, list) and any(ref in model_entries for ref in refs)
    }
else:
    kept["shadow_models"] = {}

print(f"Kept:    {len(kept)} entries")
print(f"Removed: {len(removed)} stale entries")

# Write pruned manifest back
with open(MANIFEST_PATH, "w") as f:
    json.dump(kept, f, indent=2)

print(f"Manifest written to {MANIFEST_PATH}")
