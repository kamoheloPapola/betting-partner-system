"""Phase 10 qualification and shadow collection. No promotion command."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation import phase10_current_data as phase10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','acquire','retry-statistics','qualify','qualify-all','freeze','collect','settle','evaluate','verify-phase9','finalize','verify'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--league',choices=phase10.LEAGUES)
    parser.add_argument('--attempt',default='initial')
    parser.add_argument('--source-index',type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = phase10.prepare(args.run_dir)
    elif args.command == 'qualify-all':
        result = phase10.qualify_all(args.run_dir,args.attempt)
    elif args.command in ('verify-phase9','finalize','verify'):
        from src.evaluation import phase10_evidence
        result = getattr(phase10_evidence,args.command.replace('-','_'))(args.run_dir)
    else:
        parser.error('--league is required') if args.league is None else None
        if args.command in ('freeze','collect','settle','evaluate'):
            from src.evaluation import phase10_shadow
            if args.command == 'freeze':
                result = phase10_shadow.freeze(args.run_dir,args.league,phase10.latest_qualification(args.run_dir,args.league))
            elif args.command == 'settle':
                parser.error('--source-index is required') if args.source_index is None else None
                result = phase10_shadow.settle(args.run_dir,args.league,args.source_index)
            else:
                result = getattr(phase10_shadow,args.command)(args.run_dir,args.league)
        else:
            result = getattr(phase10,args.command.replace('-','_'))(args.run_dir,args.league,args.attempt)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
