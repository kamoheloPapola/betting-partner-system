"""Owner-run only, after review, in the terminal containing the environment key."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.ingest.provider_snapshots import main
if __name__=="__main__":
    raise SystemExit(main())
