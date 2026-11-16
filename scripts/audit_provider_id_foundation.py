"""Read-only metadata-only audit. This runner does not fetch or match providers."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import traceback
from src.data_foundation.audit import run_audit,SensitiveMetadata
p=argparse.ArgumentParser()
p.add_argument("--packet",required=True)
p.add_argument("--output-leaf",default="offline-audit-01")
args=p.parse_args()
try:
    packet=Path(args.packet).resolve()
    if not args.output_leaf or Path(args.output_leaf).name!=args.output_leaf:
        raise ValueError("Unsafe output leaf")
    result=run_audit(ROOT,packet/args.output_leaf,packet)
    print(json.dumps(result))
except SensitiveMetadata:
    print("STOP: potentially sensitive metadata detected; values withheld.",file=sys.stderr)
    raise SystemExit(2)
except Exception as exc:
    print("OFFLINE_AUDIT_FAILED "+type(exc).__name__,file=sys.stderr)
    traceback.print_exc()
    raise SystemExit(1)
