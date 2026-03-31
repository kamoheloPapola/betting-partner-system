
import time
import logging
from src.features.pipeline import FeaturePipeline
import pandas as pd

# Configure logging to see our cache hits/misses
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def verify_pipeline_cache():
    logger.info("Initializing FeaturePipeline...")
    pipeline = FeaturePipeline()
    
    # 1. First Load (Cold Cache)
    logger.info("\n[1] First Load (Cold Cache)...")
    start_time = time.time()
    df1 = pipeline.run(league="PL")
    end_time = time.time()
    duration1 = end_time - start_time
    logger.info(f"   Rows: {len(df1)}")
    logger.info(f"   Time: {duration1:.4f}s")
    
    # 2. Second Load (Warm Cache)
    logger.info("\n[2] Second Load (Warm Cache)...")
    start_time = time.time()
    df2 = pipeline.run(league="PL")
    end_time = time.time()
    duration2 = end_time - start_time
    logger.info(f"   Rows: {len(df2)}")
    logger.info(f"   Time: {duration2:.4f}s")
    
    # Verification A: Object Identity (should be same object if cached)
    if df1 is df2:
        logger.info("   ✅ Object Identity confirmed (Exact same DataFrame returned)")
    else:
        logger.warning("   ❌ Objects are different (Cache might have failed or copied)")

    # Verification B: Speedup
    if duration2 < duration1 * 0.5: # Expecting massive speedup, but let's be safe
        logger.info(f"   ✅ Speedup confirmed ({duration1/duration2:.1f}x faster)")
    else:
        logger.warning(f"   ⚠️ No significant speedup (Cold: {duration1:.4f}s vs Warm: {duration2:.4f}s)")

    # 3. Third Load (Force Refresh)
    logger.info("\n[3] Third Load (Force Refresh=True)...")
    start_time = time.time()
    df3 = pipeline.run(league="PL", force_refresh=True)
    end_time = time.time()
    duration3 = end_time - start_time
    logger.info(f"   Rows: {len(df3)}")
    logger.info(f"   Time: {duration3:.4f}s")
    
    # Verification C: Identity check (Should be NEW object)
    if df3 is not df1:
        logger.info("   ✅ Refresh confirmed (New DataFrame returned)")
    else:
        logger.warning("   ❌ Object is same as cached (Refresh failed)")

    # Verification D: Integration with 'All' leagues (Global)
    logger.info("\n[4] Global Load (Cold)...")
    df_global = pipeline.run(league=None)
    logger.info(f"   Global Rows: {len(df_global)}")
    
    logger.info("\n[5] Global Load (Warm)...")
    start_time = time.time()
    df_global_2 = pipeline.run(league=None)
    logger.info(f"   Time: {time.time() - start_time:.4f}s")
    assert df_global is df_global_2, "Global cache failed identity check"

if __name__ == "__main__":
    verify_pipeline_cache()
