"""Explicit packet-only offline NEXT-36 entry point."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_foundation.root_cause import run

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--packet",required=True)
    args=parser.parse_args()
    print(run(ROOT,Path(args.packet)))
    return 0

if __name__=="__main__":
    raise SystemExit(main())

