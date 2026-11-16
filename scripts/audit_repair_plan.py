"""Packet-only offline metadata plan runner."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_foundation.repair_plan import run

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--packet",required=True)
    parser.add_argument("--output-leaf",default="")
    args=parser.parse_args()
    result=run(ROOT,Path(args.packet),args.output_leaf)
    print({k:result[k] for k in ("alias_pairs","absent_IDs","H-E","H-F","H-G","H-H","H-I")})
    return 1 if result["metadata_errors"] else 0

if __name__=="__main__":
    raise SystemExit(main())

