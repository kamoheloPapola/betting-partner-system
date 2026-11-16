"""NEXT-34 read-only metadata audit. No network, scores or model execution."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import traceback
from src.data_foundation.snapshot_metadata import run,IntegrityError,ScoreAccessError
p=argparse.ArgumentParser()
p.add_argument("--packet",required=True)
p.add_argument("--output-leaf",default="results")
args=p.parse_args()
try:
    result=run(ROOT,Path(args.packet),args.output_leaf)
    print(json.dumps(result,ensure_ascii=True),flush=True)
except IntegrityError as exc:
    print(json.dumps({"status":"STOP_INTEGRITY","details":exc.details}),file=sys.stderr)
    raise SystemExit(1)
except ScoreAccessError:
    print("STOP_SCORE_ACCESS",file=sys.stderr)
    raise SystemExit(1)
except Exception as exc:
    print("AUDIT_FAILURE "+type(exc).__name__,file=sys.stderr)
    traceback.print_exc()
    raise SystemExit(1)
