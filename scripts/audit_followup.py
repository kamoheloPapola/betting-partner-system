"""NEXT-37 packet-only preparation or one registered sandbox child."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_foundation.followup_audit import prepare,run_seed

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--packet",required=True)
    parser.add_argument("--phase",choices=("prepare","seed"),required=True)
    parser.add_argument("--variant",choices=("A","B"))
    parser.add_argument("--seed",type=int)
    parser.add_argument("--checkpoint-leaf",default="")
    args=parser.parse_args()
    if args.phase=="prepare":
        result=prepare(ROOT,Path(args.packet),args.checkpoint_leaf);print({"gate":result["gate"]["status"],"P3":result["P3"]})
    else:
        result=run_seed(ROOT,Path(args.packet),args.variant,args.seed)
        print({"variant":result["variant"],"seed":result["seed"],"calls":len(result["calls"])})
    return 0

if __name__=="__main__":raise SystemExit(main())

