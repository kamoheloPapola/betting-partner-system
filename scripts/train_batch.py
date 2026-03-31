
import subprocess
import sys
import time
from pathlib import Path

LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
MODELS = ['poisson', 'nb']

def run_training():
    print("="*60)
    print("STARTING FULL BATCH TRAINING")
    print("="*60)
    
    start_global = time.time()
    success_count = 0
    fail_count = 0
    
    for model in MODELS:
        for league in LEAGUES:
            print(f"\n[Training] Model: {model.upper()} | League: {league}")
            cmd = [
                sys.executable, "-m", "src.cli", "train", 
                model, 
                "--league", league, 
                "--mode", "production"
            ]
            
            try:
                # Capture output to avoid spamming, but print on error
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ SUCCESS ({league} {model})")
                    success_count += 1
                else:
                    print(f"❌ FAILED ({league} {model})")
                    print("Error Output:")
                    print(result.stderr)
                    print(result.stdout)
                    fail_count += 1
            except Exception as e:
                print(f"❌ EXECUTION FLIGHT ({league} {model}): {e}")
                fail_count += 1
                
    duration = time.time() - start_global
    print("\n" + "="*60)
    print(f"BATCH COMPLETE in {duration:.1f}s")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print("="*60)

if __name__ == "__main__":
    run_training()
