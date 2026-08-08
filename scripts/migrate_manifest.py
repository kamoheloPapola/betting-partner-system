
import json
from pathlib import Path

from src.config.model_state import require_unlocked

MODELS_DIR = Path("src/ml/models")
MANIFEST_FILE = MODELS_DIR / "manifest.json"

def migrate():
    require_unlocked("Manifest migration")
    if not MANIFEST_FILE.exists():
        print("No manifest found.")
        return

    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    updated_count = 0
    for key, meta in manifest.items():
        if "league" not in meta:
            meta["league"] = "PL" # Assume existing history is PL
            updated_count += 1
            print(f"Tagged {key} as PL")

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Migration complete. Updated {updated_count} models.")

if __name__ == "__main__":
    migrate()
